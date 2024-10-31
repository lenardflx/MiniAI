import argparse
import config
from src.display import Display
from src.data_loader import PDFDataLoader
from src.model import Transformer
from src.trainer import LanguageModelTrainer
from src.generator import TextGenerator
from src.cache_utils import save_model_cache, load_model_cache, is_cache_available


class MainApp:
    """
    Main application for loading data, training the model, and generating text based on a PDF document.
    """

    def __init__(self, pdf_path, start_text=None, num_epochs=500, realtime_output=False, overwrite_cache=False):
        self.pdf_path = pdf_path
        self.start_text = start_text
        self.num_epochs = num_epochs
        self.realtime_output = realtime_output
        self.overwrite_cache = overwrite_cache
        self.transformer_model = None
        self.vocab = None
        self.itos = None

    def load_or_train_model(self):
        """
        Load the model from cache if available; otherwise, load data and train a new model.
        """
        if is_cache_available(config.cache_path, self.overwrite_cache):
            # Load cached model
            self.transformer_model = Transformer(
                vocab_size=10000,  # Placeholder, adjust based on vocab size
                embed_size=config.embed_size,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                seq_len=config.seq_len
            )
            self.transformer_model.params, self.vocab, self.itos = load_model_cache(config.cache_path)
            Display.cache_loaded_message(config.cache_path)
        else:
            # Load data from PDF and initialize model
            data_loader = PDFDataLoader(self.pdf_path)
            data_loader.preprocess_data(seq_len=config.seq_len)
            data = data_loader.get_data()
            vocab_size = data_loader.get_vocab_size()
            self.vocab = data_loader.get_vocab()
            self.itos = data_loader.get_itos()

            # Initialize and train the model
            self.transformer_model = Transformer(
                vocab_size=vocab_size,
                embed_size=config.embed_size,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                seq_len=config.seq_len
            )
            trainer = LanguageModelTrainer(transformer_model=self.transformer_model, data=data, vocab_size=vocab_size)
            trainer.train(num_epochs=self.num_epochs)

            # Cache model parameters and vocabulary
            save_model_cache(self.transformer_model.params, self.vocab, self.itos, config.cache_path)
            Display.cache_saved_message(config.cache_path)

    def generate_text(self):
        """
        Generate text using the trained model, based on the provided starting text or interactive input.
        """
        generator = TextGenerator(
            model=self.transformer_model,
            params=self.transformer_model.params,
            vocab=self.vocab,
            itos=self.itos,
            realtime_output=self.realtime_output
        )

        if self.start_text:
            generated_text = generator.generate_text(self.start_text, max_len=20)
            Display.show_generated_text(generated_text)
        else:
            # Interactive loop for text generation
            print("\nTraining complete. Enter a starting text to generate text, or type 'exit' to quit.")
            while True:
                start_text = Display.prompt_start_text()
                if start_text.lower() == "exit":
                    Display.show_exit_message()
                    break
                else:
                    generated_text = generator.generate_text(start_text, max_len=20)
                    if not config.realtime_output:
                        Display.show_generated_text(generated_text)
                    else:
                        print()

    def run(self):
        """Run the main application flow: load/train model and generate text."""
        self.load_or_train_model()
        self.generate_text()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model on a PDF and generate text.")
    parser.add_argument("--pdf", default=config.pdf_path, help="Path to the PDF file")
    parser.add_argument("--start_text", help="Starting text for text generation")
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--realtime_output", action="store_true", default=config.realtime_output,
                        help="Enable real-time output during generation")
    parser.add_argument("--overwrite_cache", action="store_true", default=config.overwrite_cache,
                        help="Overwrite cached model parameters")

    args = parser.parse_args()

    app = MainApp(
        pdf_path=args.pdf,
        start_text=args.start_text,
        num_epochs=args.num_epochs,
        realtime_output=args.realtime_output,
        overwrite_cache=args.overwrite_cache
    )
    app.run()
