import jax
import jax.numpy as jnp

class Transformer:
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, seq_len):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Initialize model parameters
        self.params = self.init_params()

    def init_params(self):
        """Initialize parameters for the transformer model."""
        key = jax.random.PRNGKey(0)
        params = {}
        params['embedding'] = jax.random.normal(key, (self.vocab_size, self.embed_size))
        params['positional_encoding'] = self.create_positional_encoding()
        params['layers'] = []
        for _ in range(self.num_layers):
            layer_params = {
                'qkv': jax.random.normal(key, (self.num_heads, self.embed_size, self.embed_size // self.num_heads)),
                'ffn1': jax.random.normal(key, (self.embed_size, self.hidden_dim)),  # First feed-forward layer
                'ffn2': jax.random.normal(key, (self.hidden_dim, self.embed_size)),  # Second feed-forward layer
                'output': jax.random.normal(key, (self.embed_size, self.vocab_size)),
            }
            params['layers'].append(layer_params)
        return params

    def create_positional_encoding(self, max_len=512):
        pos = jnp.arange(max_len)[:, None]
        i = jnp.arange(0, self.embed_size, 2)
        angle_rates = jnp.exp(-i * jnp.log(10000.0) / self.embed_size)
        pos_encoding = jnp.zeros((max_len, self.embed_size))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(pos * angle_rates))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(pos * angle_rates))
        return pos_encoding

    def forward(self, x, mask):
        """Apply embeddings, positional encoding, and transformer layers with masking."""
        # Ensure x has two dimensions: (batch_size, seq_len)
        if x.ndim == 1:
            x = x.reshape(1, -1)  # Reshape to add a batch dimension if necessary

        # Embed the input and add positional encoding
        x = self.embed(x)
        batch_size, seq_len, embed_size = x.shape
        pos_encoding = self.params['positional_encoding'][:seq_len, :].reshape(1, seq_len, embed_size)
        x += jnp.broadcast_to(pos_encoding,
                              x.shape)  # Broadcast positional encoding to (batch_size, seq_len, embed_size)

        # Apply transformer layers
        for layer in self.params['layers']:
            # Multi-head attention
            x = self.multi_head_attention(x, layer, mask) + x

            # Feed-forward network with two layers and residual connection
            x_ffn = jax.nn.relu(jnp.dot(x, layer['ffn1']))
            x = jnp.dot(x_ffn, layer['ffn2']) + x  # Residual connection

        # Final output projection
        logits = jnp.dot(x, self.params['layers'][-1]['output'])
        return logits

    def embed(self, x):
        return jnp.take(self.params['embedding'], x, axis=0)

    def multi_head_attention(self, x, layer_params, mask):
        """Compute multi-head self-attention with masking support."""
        # Ensure x has shape (batch_size, seq_len, embed_size) before multi-head attention
        batch_size, seq_len, embed_size = x.shape
        num_heads = self.num_heads
        d_k = embed_size // num_heads  # Dimension per head

        # Linear projections for query, key, and value
        qkv = jnp.einsum('bld,hde->bhle', x, layer_params['qkv'])  # Shape: (batch_size, num_heads, seq_len, d_k)
        scores = jnp.einsum('bhld,bhmd->bhlm', qkv, qkv) / jnp.sqrt(
            d_k)  # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Adjust mask shape to match scores
        if mask.ndim == 2:  # If mask is (batch_size, seq_len)
            mask = mask[:, None, None, :]  # Reshape to (batch_size, 1, 1, seq_len) for broadcasting
        elif mask.ndim == 3:  # If mask is (batch_size, seq_len, seq_len)
            mask = mask[:, None, :, :]  # Reshape to (batch_size, 1, seq_len, seq_len)

        # Apply mask to scores
        scores = jnp.where(mask, scores, -1e9)  # Masked positions receive a large negative value

        # Compute attention weights
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = jnp.einsum('bhlm,bhmd->bhld', attention_weights,
                                 qkv)  # Shape: (batch_size, num_heads, seq_len, d_k)

        # Reshape attn_output back to (batch_size, seq_len, embed_size)
        return jnp.reshape(attn_output, (batch_size, seq_len, embed_size))
