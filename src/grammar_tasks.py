import jax.numpy as jnp
import random


class GrammarTaskError(Exception):
    """Exception for handling grammar task-related errors."""
    pass


class GrammarTasks:
    """
    Handles grammar-aware tasks for training, such as masked language modeling
    and sequence reordering.
    """

    def __init__(self, mask_token_id=1, mask_prob=0.15):
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

    def masked_language_modeling(self, sentence):
        """Applies masking to random tokens in a sentence."""
        try:
            return jnp.array([
                self.mask_token_id if random.random() < self.mask_prob else token
                for token in sentence
            ])
        except Exception as e:
            raise GrammarTaskError("Error in masked language modeling.") from e

    def sequence_reordering(self, sentence, chunk_size=2):
        """Shuffles chunks within a sentence for a reordering task."""
        try:
            chunks = [sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)]
            random.shuffle(chunks)
            shuffled_sentence = [token for chunk in chunks for token in chunk]
            return jnp.array(shuffled_sentence), jnp.array(sentence)
        except Exception as e:
            raise GrammarTaskError("Error in sequence reordering task.") from e
