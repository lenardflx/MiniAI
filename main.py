import argparse
import config
from jax import random
from src.data_loader import PDFDataLoader
from src.model import Transformer
from src.trainer import LanguageModelTrainer
from src.generator import TextGenerator

def main(pdf_path=config.pdf_path, start_text="", num_epochs=config.num_epochs,
         realtime_output=config.realtime_output, temperature=1.0, top_k=10):
    # Load and preprocess data from PDF
    data_loader = PDFDataLoader(pdf_path)
    data_loader.preprocess_data(seq_len=config.seq_len)
    data = data_loader.get_data()
    vocab_size = data_loader.get_vocab_size()
    vocab = data_loader.get_vocab()
    itos = data_loader.get_itos()

    # Initialize model with parameters from config.py
    transformer_model = Transformer(
        vocab_size=vocab_size,
        embed_size=config.embed_size,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        seq_len=config.seq_len
    )

    # Train the model
    trainer = LanguageModelTrainer(
        transformer_model=transformer_model,
        data=data,
        vocab_size=vocab_size,
        learning_rate=config.learning_rate,
        label_smoothing=config.label_smoothing,
        clip_value=config.clip_value
    )

    print("Starting training...")
    trainer.train(num_epochs=num_epochs)

    # Initialize TextGenerator with trained model parameters and sampling controls
    generator = TextGenerator(
        model=transformer_model,
        params=transformer_model.params,
        vocab=vocab,
        itos=itos,
        realtime_output=realtime_output,
        temperature=temperature,
        top_k=top_k
    )

    # If start_text is provided, generate text once; otherwise, enter input loop
    if start_text:
        generated_text = generator.generate_text(start_text, max_len=20)
        print(f"\nGenerated Text: {generated_text}\n")
    else:
        print("\nTraining complete. Enter a starting text to generate text, or type 'exit' to quit.")
        while True:
            start_text = input("Enter starting text (or type 'exit' to quit): ")
            if start_text.lower() == "exit":
                print("Exiting text generation.")
                break
            else:
                generated_text = generator.generate_text(start_text, max_len=20)
                if not config.realtime_output:
                    print(f"\nGenerated Text: {generated_text}\n")
                else:
                    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model on a PDF and generate text.")
    parser.add_argument("--pdf", default=config.pdf_path, help="Path to the PDF file")
    parser.add_argument("--start_text", help="Starting text for text generation")
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--realtime_output", action="store_true", default=config.realtime_output,
                        help="Enable real-time output during generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for text generation (controls randomness)")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling for text generation (controls diversity)")
    args = parser.parse_args()

    # Run main with either command-line arguments or defaults from config.py
    main(
        pdf_path=args.pdf,
        start_text=args.start_text,
        num_epochs=args.num_epochs,
        realtime_output=args.realtime_output,
        temperature=args.temperature,
        top_k=args.top_k
    )
