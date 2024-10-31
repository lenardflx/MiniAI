# config.py

# Path to the default PDF file
pdf_path = "example/data.pdf"

# Model training configurations
num_epochs = 10               # Increased epochs for extensive training
batch_size = 16               # Moderate batch size to stabilize learning with high resource utilization
learning_rate = 0.0003        # Lower learning rate for more controlled updates
realtime_output = True        # Display output in real-time during generation

# Model architecture parameters for improved quality
seq_len = 40                  # Increased sequence length to capture larger context
embed_size = 240              # Larger embedding size for richer token representation => divisible by num_heads
num_heads = 12                # Increase number of attention heads for nuanced attention
hidden_dim = 512              # Larger hidden dimension for handling more complex transformations
num_layers = 8                # Deeper model for better learning of language intricacies

# Text generation parameters
temperature = 0.7             # Slightly lower temperature for coherent and focused generation
top_k = 10                    # Top-k sampling for high-quality, contextually relevant tokens
label_smoothing = 0.05        # Small label smoothing for improved generalization
clip_value = 1.0              # Gradient clipping to ensure stable training

