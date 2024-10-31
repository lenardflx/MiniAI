import jax
import jax.numpy as jnp


class Transformer:
    """
    A transformer-based model for sequence tasks with self-attention and feed-forward layers.
    """

    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, seq_len):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.params = self.init_params()

    def init_params(self):
        """Initializes model parameters, including embedding and transformer layer weights."""
        key = jax.random.PRNGKey(0)
        params = {'embedding': jax.random.normal(key, (self.vocab_size, self.embed_size)),
                  'positional_encoding': self.create_positional_encoding()}
        params['layers'] = [
            {'qkv': jax.random.normal(key, (self.num_heads, self.embed_size, self.embed_size // self.num_heads)),
             'ffn1': jax.random.normal(key, (self.embed_size, self.hidden_dim)),
             'ffn2': jax.random.normal(key, (self.hidden_dim, self.embed_size)),
             'output': jax.random.normal(key, (self.embed_size, self.vocab_size))}
            for _ in range(self.num_layers)]
        return params

    def create_positional_encoding(self, max_len=512):
        """Generates positional encoding to add sequential information to embeddings."""
        pos = jnp.arange(max_len)[:, None]
        i = jnp.arange(0, self.embed_size, 2)
        angle_rates = jnp.exp(-i * jnp.log(10000.0) / self.embed_size)
        pos_encoding = jnp.zeros((max_len, self.embed_size))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(pos * angle_rates))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(pos * angle_rates))
        return pos_encoding

    def forward(self, x, mask):
        """Performs forward pass through embedding, positional encoding, and transformer layers."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Embed the input sequence
        x = self.embed(x)

        # Broadcast positional encoding to match (batch_size, seq_len, embed_size)
        batch_size, seq_len, embed_size = x.shape
        pos_encoding = self.params['positional_encoding'][:seq_len].reshape(1, seq_len, embed_size)
        x += jnp.broadcast_to(pos_encoding, x.shape)  # Broadcasting positional encoding

        # Apply transformer layers
        for layer in self.params['layers']:
            x = self.multi_head_attention(x, layer, mask) + x
            x = jnp.dot(jax.nn.relu(jnp.dot(x, layer['ffn1'])), layer['ffn2']) + x

        # Project to output logits
        return jnp.dot(x, self.params['layers'][-1]['output'])

    def embed(self, x):
        """Embed input sequence tokens."""
        return jnp.take(self.params['embedding'], x, axis=0)

    def multi_head_attention(self, x, layer_params, mask):
        """Applies multi-head self-attention with optional masking."""
        batch_size, seq_len, _ = x.shape
        d_k = self.embed_size // self.num_heads
        qkv = jnp.einsum('bld,hde->bhle', x, layer_params['qkv'])
        scores = jnp.einsum('bhld,bhmd->bhlm', qkv, qkv) / jnp.sqrt(d_k)
        scores = jnp.where(mask, scores, -1e9)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.einsum('bhlm,bhmd->bhld', attention_weights, qkv)
        return jnp.reshape(attn_output, (batch_size, seq_len, self.embed_size))
