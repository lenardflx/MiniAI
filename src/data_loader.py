import fitz  # PyMuPDF for reading PDF files
from collections import defaultdict


class DataLoadingError(Exception):
    """Custom exception for errors during data loading or preprocessing."""
    pass


class PDFDataLoader:
    """
    PDFDataLoader handles text extraction, tokenization, and vocabulary building from a PDF file.
    """

    def __init__(self, file_path, vocab_size=5000):
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.word_to_id = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
        self.id_to_word = {0: "<unk>", 1: "<pad>", 2: "<eos>"}
        self.data = None

    def extract_text(self):
        """Extracts and returns text content from a PDF file."""
        try:
            text = ""
            with fitz.open(self.file_path) as doc:
                for page in doc:
                    text += page.get_text("text")
            return text
        except Exception as e:
            raise DataLoadingError("Failed to extract text from PDF.") from e

    def tokenize(self, text):
        """Tokenizes text by splitting on spaces and punctuations."""
        return text.lower().replace(".", " .").split()

    def build_vocab(self, tokenized_data):
        """Builds a vocabulary dictionary from the tokenized text data."""
        try:
            word_freq = defaultdict(int)
            for word in tokenized_data:
                word_freq[word] += 1
            sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)
            for idx, word in enumerate(sorted_words[:self.vocab_size - 3], start=3):
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
        except Exception as e:
            raise DataLoadingError("Error in building vocabulary.") from e

    def preprocess_data(self, seq_len=10):
        """
        Processes data from PDF to tokenized sequences with padding. Converts text to a list of token IDs.
        """
        try:
            text = self.extract_text()
            tokenized_data = self.tokenize(text)
            self.build_vocab(tokenized_data)

            # Convert text to token IDs and chunk into sequences of seq_len
            token_ids = [self.word_to_id.get(word, self.word_to_id["<unk>"]) for word in tokenized_data]
            token_ids.append(self.word_to_id["<eos>"])

            # Chunk token IDs into sequences for batching
            self.data = [token_ids[i:i + seq_len] for i in range(0, len(token_ids), seq_len)]
            return self.data
        except DataLoadingError as e:
            print(e)

    def get_data(self):
        """Returns preprocessed data for training."""
        if self.data is None:
            raise DataLoadingError("Data has not been preprocessed. Call preprocess_data first.")
        return self.data

    def get_vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.word_to_id)

    def get_vocab(self):
        """Returns word-to-ID mapping."""
        return self.word_to_id

    def get_itos(self):
        """Returns ID-to-word mapping for decoding."""
        return self.id_to_word
