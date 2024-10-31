# grammar_tasks.py
import jax.numpy as jnp
import random

class GrammarTasks:
    def __init__(self, mask_token_id=1, mask_prob=0.15):
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

    def masked_language_modeling(self, sentence):
        """Apply masked language modeling by masking random tokens."""
        masked_sentence = [
            self.mask_token_id if random.random() < self.mask_prob else token
            for token in sentence
        ]
        return jnp.array(masked_sentence)

    def sequence_reordering(self, sentence, chunk_size=2):
        """Shuffle chunks within the sentence and prepare a reordering task."""
        chunks = [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]
        random.shuffle(chunks)
        shuffled_sentence = [token for chunk in chunks for token in chunk]
        return jnp.array(shuffled_sentence), jnp.array(sentence)  # Return shuffled and target sentence
