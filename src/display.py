class Display:
    """
    Display class to handle user interaction and output.
    This keeps display logic separate from processing, enabling cleaner modularity and reusability.
    """

    @staticmethod
    def _generate_progress_bar(progress,length=20):
        return f"[{'#' * int(length*progress)}{' ' * (length - int(length*progress))}]"

    @staticmethod
    def show_generated_text(generated_text):
        print(f"\nGenerated Text: {generated_text}\n")

    @staticmethod
    def show_exit_message():
        print("Exiting text generation.")

    @staticmethod
    def prompt_start_text():
        return input("Enter starting text (or type 'exit' to quit): ")

    @staticmethod
    def training_progress(epoch, num_epochs, progress, avg_loss, time_remaining):
        """Show training progress."""
        print(
            f"\rEpoch {epoch + 1}/{num_epochs} {Display._generate_progress_bar(progress)} "
            f"{progress * 100:.2f}% complete - Avg Loss: {avg_loss} - "
            f"Time remaining: {time_remaining}",
            end=""
        )

    @staticmethod
    def training_complete_message(duration):
        print(f"\nTraining completed in {duration}")

    @staticmethod
    def cache_saved_message(cache_path):
        print(f"Model parameters and vocabulary saved to cache: {cache_path}")

    @staticmethod
    def cache_loaded_message(cache_path):
        print(f"Loaded model parameters and vocabulary from cache: {cache_path}")
