import jax
import jax.numpy as jnp
import random


class TextGenerator:
    def __init__(self, model, params, vocab, itos, realtime_output=False, temperature=1.0, top_k=10):
        """Initialize the text generator with model, params, vocab, and additional options."""
        self.model = model
        self.params = params
        self.vocab = vocab
        self.itos = itos
        self.realtime_output = realtime_output
        self.temperature = max(temperature, 1e-2)  # Minimum temperature for stability
        self.top_k = top_k

    def sample_logits(self, logits):
        """Apply temperature scaling and top-k sampling to logits with enhanced stability."""
        # Initial clipping to prevent very large values
        logits = jnp.clip(logits, a_min=-1e9, a_max=1e9)

        # Scale logits by temperature, then clip again
        logits = logits / self.temperature
        logits = jnp.clip(logits, a_min=-1e9, a_max=1e9)

        # Get top-k probabilities
        top_k_indices = jnp.argsort(logits)[-self.top_k:]  # Get indices of top k elements
        top_k_logits = logits[top_k_indices]

        # Convert logits to probabilities with small epsilon for stability
        top_k_probs = jax.nn.softmax(top_k_logits + 1e-8)

        # Check if probabilities are finite
        if not jnp.isfinite(top_k_probs).all():
            print("Debug Info - logits:", logits)
            print("Debug Info - top_k_logits:", top_k_logits)
            raise ValueError("Probabilities contain non-finite values. Check logits and temperature scaling.")

        # Sample from the top-k probabilities
        choice = random.choices(top_k_indices, weights=top_k_probs, k=1)
        return int(choice[0])

    def generate_text(self, start_text, max_len=20):
        """Generate text from the starting prompt using the trained model."""
        start_tokens = [self.vocab.get(token, self.vocab["<unk>"]) for token in start_text.split()]
        generated_tokens = list(start_tokens)

        for _ in range(max_len):
            input_seq = jnp.array(generated_tokens).reshape(1, -1)
            mask = (input_seq != self.vocab["<pad>"]).astype(jnp.float32).reshape(1, 1, 1, -1)

            # Get output logits from the model
            logits = self.model.forward(input_seq, mask)[0, -1, :]  # Last token logits

            # Sample the next token using temperature scaling and top-k sampling
            next_token = self.sample_logits(logits)
            generated_tokens.append(next_token)

            if self.realtime_output:
                print(self.itos[next_token], end=" ", flush=True)

            # Stop if end of sequence token is generated
            if next_token == self.vocab["<eos>"]:
                break

        generated_text = " ".join([self.itos[int(token)] for token in generated_tokens[len(start_tokens):]])
        return generated_text
