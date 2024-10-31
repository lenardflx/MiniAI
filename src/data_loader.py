import fitz  # PyMuPDF for reading PDF files
import jax.numpy as jnp
from collections import defaultdict

class PDFDataLoader:
    def __init__(self, file_path, vocab_size=5000):
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.word_to_id = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
        self.id_to_word = {0: "<unk>", 1: "<pad>", 2: "<eos>"}
        self.data = None

    def extract_text(self):
        """Extract text from the PDF file."""
        text = ""
        with fitz.open(self.file_path) as doc:
            for page in doc:
                text += page.get_text("text")
        return text

    def tokenize(self, text):
        """Tokenize the text into a list of words."""
        return text.lower().replace(".", " .").split()

    def build_vocab(self, tokenized_data):
        """Build a vocabulary from the tokenized text data."""
        word_freq = defaultdict(int)
        for word in tokenized_data:
            word_freq[word] += 1
        sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)
        for idx, word in enumerate(sorted_words[:self.vocab_size - 3], start=3):  # Reserving 0, 1, 2 for special tokens
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    def preprocess_data(self, seq_len=10):
        """Extract, tokenize, and convert text to a list of padded token IDs for training."""
        text = self.extract_text()
        tokenized_data = self.tokenize(text)
        self.build_vocab(tokenized_data)

        # Convert text to token IDs and chunk into sequences of seq_len
        token_ids = [self.word_to_id.get(word, self.word_to_id["<unk>"]) for word in tokenized_data]
        token_ids.append(self.word_to_id["<eos>"])  # Add EOS token

        # Break token IDs into chunks of seq_len for batching
        self.data = [token_ids[i:i + seq_len] for i in range(0, len(token_ids), seq_len)]
        return self.data

    def get_data(self):
        """Return the preprocessed tokenized data."""
        if self.data is None:
            raise ValueError("Data has not been preprocessed. Call preprocess_data first.")
        return self.data

    def get_vocab_size(self):
        """Return the size of the vocabulary."""
        return len(self.word_to_id)

    def get_vocab(self):
        """Return the word-to-ID dictionary for the vocabulary."""
        return self.word_to_id

    def get_itos(self):
        """Return the ID-to-word dictionary for decoding generated text."""
        return self.id_to_word
