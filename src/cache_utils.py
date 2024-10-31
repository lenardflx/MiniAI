import os
import pickle


class CacheError(Exception):
    """Custom exception for handling cache-related errors."""
    pass


def save_model_cache(model_params, vocab, itos, cache_path):
    """Save model parameters and vocabulary to cache file."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {'model_params': model_params, 'vocab': vocab, 'itos': itos}
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"Model cache saved to {cache_path}.")
    except Exception as e:
        raise CacheError("Failed to save model cache.") from e


def load_model_cache(cache_path):
    """Load model parameters and vocabulary from cache file."""
    try:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        print(f"Model cache loaded from {cache_path}.")
        return cache_data['model_params'], cache_data['vocab'], cache_data['itos']
    except FileNotFoundError:
        raise CacheError("Cache file not found.")
    except Exception as e:
        raise CacheError("Error loading model cache.") from e


def is_cache_available(cache_path, overwrite_cache):
    """Check if cache is available based on path and overwrite flag."""
    return os.path.exists(cache_path) and not overwrite_cache
