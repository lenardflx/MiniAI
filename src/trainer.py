import random
import time
import jax
import jax.numpy as jnp
import optax
from src.display import Display
from src.grammar_tasks import GrammarTasks


class TrainingError(Exception):
    """Custom exception class for handling training errors."""
    pass


class LanguageModelTrainer:
    """
    Language model trainer that manages the training loop, optimizer updates,
    and optional grammar tasks for training the language model.
    """

    def __init__(self, transformer_model, data, vocab_size, learning_rate=0.001):
        self.transformer_model = transformer_model
        self.data = data
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.grammar_tasks = GrammarTasks(mask_token_id=1, mask_prob=0.15)

        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.transformer_model.params)

    def create_mask(self, x):
        """
        Create a mask for padding tokens in the input sequence to prevent
        unnecessary gradient updates.
        """
        try:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            return (x != 1).astype(jnp.float32)[:, None, None, :]
        except Exception as e:
            raise TrainingError("Error in creating mask.") from e

    def loss_fn(self, params, x, y, mask):
        """Compute the masked cross-entropy loss between predicted and target tokens."""
        try:
            logits = self.transformer_model.forward(x, mask)
            logits = jnp.clip(logits, a_min=-1e9, a_max=1e9)
            one_hot_labels = jax.nn.one_hot(y, num_classes=self.vocab_size)
            epsilon = 1e-8
            return -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits + epsilon)) / mask.sum()
        except Exception as e:
            raise e #TrainingError("Error in computing loss.") from e

    def grammar_aware_loss_fn(self, params, x, y, mask, task_type):
        """Compute the loss for the specified grammar-aware task."""
        if task_type == 'MLM':
            logits = self.transformer_model.forward(x, mask)
            logits = jnp.clip(logits, a_min=-1e9, a_max=1e9)  # Stability
            one_hot_labels = jax.nn.one_hot(y, num_classes=self.vocab_size)
            loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits)) / mask.sum()
        elif task_type == 'Reorder':
            logits = self.transformer_model.forward(x, mask)
            logits = jnp.clip(logits, a_min=-1e9, a_max=1e9)
            one_hot_labels = jax.nn.one_hot(y, num_classes=self.vocab_size)
            loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits)) / mask.sum()
        else:
            raise ValueError("Unknown task type")
        return loss

    def train_step(self, params, opt_state, x, y, mask, task_type='Standard'):
        """Perform a single training step, with optional grammar tasks."""
        # Choose task type randomly for diversity
        if task_type == 'Standard':
            loss_fn = self.loss_fn  # Standard language modeling loss
        else:
            loss_fn = lambda p, x, y, m: self.grammar_aware_loss_fn(p, x, y, m, task_type)

        grads = jax.grad(loss_fn)(params, x, y, mask)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def train(self, num_epochs=500):
        """
        Conducts the training loop with periodic display of progress and
        estimated time remaining.
        """
        try:
            start_time = time.time()
            total_batches = len(self.data) * num_epochs
            current_batch = 0

            for epoch in range(num_epochs):
                for i, sentence in enumerate(self.data):
                    x = jnp.array(sentence[:-1])
                    y = jnp.array(sentence[1:])
                    mask = self.create_mask(x)
                    task_type = random.choice(['Standard', 'MLM', 'Reorder'])

                    if task_type == 'MLM':
                        x = self.grammar_tasks.masked_language_modeling(sentence[:-1])
                    elif task_type == 'Reorder':
                        x, y = self.grammar_tasks.sequence_reordering(sentence[:-1])

                    self.transformer_model.params, self.opt_state = self.train_step(
                        self.transformer_model.params, self.opt_state, x, y, mask, task_type
                    )

                    # Progress calculation and output
                    current_batch += 1
                    progress = current_batch / total_batches
                    elapsed_time = time.time() - start_time
                    estimated_time_remaining = self.format_time(elapsed_time / progress - elapsed_time)
                    avg_loss = self.loss_fn(self.transformer_model.params, x, y, mask)

                    if current_batch % 10 == 0:
                        Display.training_progress(epoch, num_epochs, progress, avg_loss, estimated_time_remaining)

            Display.training_complete_message(self.format_time(time.time() - start_time))
        except TrainingError as e:
            print(f"Training failed: {e}")

    @staticmethod
    def format_time(seconds):
        """Format seconds into mm:ss format for time estimation."""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02}:{seconds:02}"
