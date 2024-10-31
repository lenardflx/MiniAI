import random
import time
from src.grammar_tasks import GrammarTasks
import jax
import jax.numpy as jnp
import optax

class LanguageModelTrainer:
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
        """Create a mask for padding tokens in the input sequence."""
        # Check if x has a batch dimension and is a 2D array
        if x.ndim == 1:
            x = x.reshape(1, -1)  # Ensure x is (batch_size, seq_len)

        # Create a mask where 1 represents valid tokens and 0 represents padding tokens
        mask = (x != 1).astype(jnp.float32)  # Assuming padding token ID is 1

        # Reshape mask to (batch_size, 1, 1, seq_len) for broadcasting
        mask = mask[:, None, None, :]  # Adds the necessary dimensions for broadcasting

        return mask

    def loss_fn(self, params, x, y, mask):
        """Compute the loss for given input and target with masking, preventing NaN."""
        logits = self.transformer_model.forward(x, mask)

        # Stabilize logits by clipping to avoid NaN values
        logits = jnp.clip(logits, a_min=-1e9, a_max=1e9)

        # Calculate loss with added epsilon for stability
        one_hot_labels = jax.nn.one_hot(y, num_classes=self.vocab_size)
        epsilon = 1e-8  # Small value to avoid log(0)
        loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits + epsilon)) / mask.sum()
        return loss

    def format_loss(self, loss):
        """Format the loss with suffixes for thousands (k), millions (M), etc."""
        if loss >= 1e6:
            return f"{loss / 1e6:.2f}M"
        elif loss >= 1e3:
            return f"{loss / 1e3:.2f}k"
        else:
            return f"{loss:.2f}"

    def format_time(self, seconds):
        """Convert time in seconds to mm:ss format."""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02}:{seconds:02}"

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
        total_batches = len(self.data) * num_epochs
        current_batch = 0
        start_time = time.time()
        bar_length = 20  # Length of the progress bar

        for epoch in range(num_epochs):
            total_loss = 0
            for i, sentence in enumerate(self.data):
                x = jnp.array(sentence[:-1])  # Input tokens
                y = jnp.array(sentence[1:])  # Target tokens
                mask = self.create_mask(x)

                # Select task type randomly
                task_type = random.choice(['Standard', 'MLM', 'Reorder'])

                # Prepare task data
                if task_type == 'MLM':
                    x = self.grammar_tasks.masked_language_modeling(sentence[:-1])
                elif task_type == 'Reorder':
                    x, y = self.grammar_tasks.sequence_reordering(sentence[:-1])

                # Perform a training step
                self.transformer_model.params, self.opt_state = self.train_step(
                    self.transformer_model.params, self.opt_state, x, y, mask, task_type
                )

                # Calculate loss for reporting
                loss = self.loss_fn(self.transformer_model.params, x, y, mask)
                total_loss += loss

                # Update batch count and display progress
                current_batch += 1
                percent_complete = (current_batch / total_batches) * 100
                num_hashes = int((percent_complete / 100) * bar_length)
                progress_bar = f"[{'#' * num_hashes}{' ' * (bar_length - num_hashes)}]"
                elapsed_time = time.time() - start_time
                avg_batch_time = elapsed_time / current_batch
                estimated_time_remaining = self.format_time(avg_batch_time * (total_batches - current_batch))
                avg_loss = total_loss / current_batch

                # Display progress every 10 batches
                if current_batch % 10 == 0:
                    print(
                        f"\rEpoch {epoch + 1}/{num_epochs} {progress_bar} "
                        f"{percent_complete:.2f}% complete - "
                        f"Avg Loss: {self.format_loss(avg_loss)} - "
                        f"Time remaining: {estimated_time_remaining}",
                        end=""
                    )

        print(f"\rEpoch {num_epochs}/{num_epochs} [###################] {100:.2f}% completed training")
