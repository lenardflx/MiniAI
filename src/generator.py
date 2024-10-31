import jax
import jax.numpy as jnp
import random


class TextGenerationError(Exception):
    """Custom exception for errors in text generation."""
    pass


class TextGenerator:
    """
    Text generator using a trained Transformer model to generate text based on a starting prompt.
    """

    def __init__(self, model, params, vocab, itos, realtime_output=False, temperature=1.0, top_k=10):
        self.model = model
        self.params = params
        self.vocab = vocab
        self.itos = itos
        self.realtime_output = realtime_output
        self.temperature = max(temperature, 1e-2)
        self.top_k = top_k

    def sample_logits(self, logits):
        """Sample the next token from the model's logits using top-k sampling and temperature scaling."""
        try:
            logits = jnp.clip(logits, a_min=-1e9, a_max=1e9) / self.temperature
            top_k_indices = jnp.argsort(logits)[-self.top_k:]
            top_k_probs = jax.nn.softmax(logits[top_k_indices] + 1e-8)

            if not jnp.isfinite(top_k_probs).all():
                raise TextGenerationError("Non-finite probabilities detected.")

            return int(random.choices(top_k_indices, weights=top_k_probs, k=1)[0])
        except Exception as e:
            raise TextGenerationError("Error in sampling logits.") from e

    def generate_text(self, start_text, max_len=20):
        """
        Generate text starting from a prompt using the model and specified sampling strategy.
        """
        try:
            start_tokens = [self.vocab.get(token, self.vocab["<unk>"]) for token in start_text.split()]
            generated_tokens = list(start_tokens)

            if self.realtime_output:
                print(start_text, end=" ", flush=True)

            for _ in range(max_len):
                input_seq = jnp.array(generated_tokens).reshape(1, -1)
                mask = (input_seq != self.vocab["<pad>"]).astype(jnp.float32).reshape(1, 1, 1, -1)
                logits = self.model.forward(input_seq, mask)[0, -1, :]
                next_token = self.sample_logits(logits)
                generated_tokens.append(next_token)

                if next_token == self.vocab["<eos>"]:
                    break
                if self.realtime_output:
                    print(self.itos[next_token], end=" ", flush=True)

            return " ".join(self.itos[token] for token in generated_tokens[len(start_tokens):])
        except TextGenerationError as e:
            print(f"Text generation failed: {e}")
