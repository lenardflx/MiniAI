# Default configuration for main.py

# Path to the default PDF file
pdf_path = "example/data.pdf"

# Model training configurations
num_epochs = 2               # Increase epochs for better learning
realtime_output = True         # Display output in real-time during generation

# Model architecture parameters for improved quality
seq_len = 30                  # Increased sequence length for longer context
embed_size = 128              # Larger embedding size for better representation
num_heads = 8                 # More heads to increase model's attention capacity
hidden_dim = 256              # Larger hidden dimension for more complex transformations
num_layers = 5                # Increased number of layers for deeper learning
learning_rate=0.001

# Text generation parameters
temperature = 0.9             # Lower temperature for more focused generation
top_k = 5                     # Top-k sampling for increased coherence in generated text
