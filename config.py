# Default configuration for MiniAI
"""
Configuration settings for the language model application.
Modify these values as needed for training, model parameters, and caching.
"""

# === File Paths ===
pdf_path = "example/main.pdf"  # Path to the folder containing text data files (PDFs or text)
cache_path = "model_cache/model_params.pkl" # Path for the model cache to store the trained model

# === Training Configurations ===
num_epochs = 100            # 1–10 for testing, 10–50 for full training; higher values = longer training
realtime_output = True    # set to True for interactive output, similar to ChatGPT
overwrite_cache = False   # set to True to always train the AI when running

# === Model Architecture Parameters ===
seq_len = 30              # Sequence length in tokens, recommended: 10–50
embed_size = 256          # Embedding vector size, recommended: 64–512
num_heads = 16             # Number of attention heads, must divide evenly into embed_size (e.g., 4, 8, 16)
hidden_dim = 256          # Hidden layer size in FFN, recommended: 128–1024
num_layers = 5            # Number of transformer layers, recommended: 2–12

# === Text Generation Parameters ===
temperature = 0.7         # Controls randomness, 0.5–1.0; lower = more focused, higher = more random
top_k = 4                 # Top-k sampling, recommended: 1–20; lower = more coherent, higher = more random

# === Notes ===
# - Ensure that `num_heads` divides `embed_size` evenly to avoid shape mismatch errors.
# - Increasing `seq_len`, `embed_size`, or `num_layers` will require more memory.
# - Setting `temperature` > 1.0 or `top_k` > 20 may reduce coherence in generated text.
