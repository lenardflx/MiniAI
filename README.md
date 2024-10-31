# MiniAI
>PDF-Powered Text Generator with Transformers

MiniAI is a Python-based project that trains a transformer model on text extracted from a PDF, then generates text based on user prompts in the style and structure of the training data.
It’s a great way to explore how language modeling, transformers, and text generation of big AI companies works in a small prototype.

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [How to Use MiniAI](#how-to-use-miniai)
4. [Configuration](#configuration)
5. [Under the Hood](#under-the-hood)

---

## Overview

MiniAI’s transformer model is trained using data extracted from any PDF you provide, allowing it to generate contextually relevant text based on what it’s learned.
The model includes caching for trained parameters, so you don’t need to re-train every time you want to generate text.

## Getting Started

### Prerequisites

To get started, ensure you have:
- **Python 3.8+** installed
- Libraries like **JAX**, **Optax**, and **PyMuPDF** for model training and PDF handling
  - skip if installing via requirements.txt

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/lenardflx/MiniAI.git
   cd MiniAI
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```


---

## How to Use MiniAI

MiniAI can be run from the command line or by using the default settings in `config.py`.

### Running with Command-Line Options

The PDF file is required for training the model. If a starting text isn’t provided, MiniAI will prompt you in a loop to enter text for generating content.

#### Command-Line Arguments

| Parameter         | Type      | Description                                                                                                   |
|-------------------|-----------|---------------------------------------------------------------------------------------------------------------|
| `--pdf`           | `string`  | **Required.** Path to the PDF file containing the text for training.                                          |
| `--start_text`    | `string`  | Starting prompt for text generation. If omitted, the AI enters a prompt loop for text generation.            |
| `--num_epochs`    | `integer` | Number of epochs for training. Higher values lead to better results but take longer.                          |
| `--realtime_output` | `boolean` | If set, generates text word-by-word, displaying each word in real-time.                                      |
| `--overwrite_cache` | `boolean` | If set, the model ignores any cached data and trains anew.                                                   |

#### Example Usage

```bash
python main.py --pdf path/to/yourfile.pdf --start_text "Once upon a time" --num_epochs 5 --realtime_output
```

### Running with Default Configuration

If you want to skip the command-line arguments, you can set up your configurations directly in `config.py` and simply run the model as:

```bash
python main.py
```

---

## Configuration

The `config.py` file provides various settings that you can modify to control model behavior, training parameters, and text generation options.
This is the default file the model reads if no command-line arguments are given.

```python
# Default configuration for MiniAI
"""
Configuration settings for the language model application.
Modify these values as needed for training, model parameters, and caching.
"""

# === File Paths ===
pdf_path = "example/main.pdf"  # Path to the PDF file containing text data
cache_path = "model_cache/model_params.pkl"  # Location to save/load the model cache

# === Training Configurations ===
num_epochs = 100          # Training iterations; higher values = better performance, but longer runtime
realtime_output = True    # Enables interactive, real-time output, like ChatGPT
overwrite_cache = False   # Forces retraining even if a cached model exists

# === Model Architecture Parameters ===
seq_len = 30              # Token sequence length for training; recommended: 10–50
embed_size = 256          # Embedding size; recommended: 64–512
num_heads = 16            # Number of attention heads; must divide evenly into `embed_size`
hidden_dim = 256          # Size of feed-forward network; recommended: 128–1024
num_layers = 5            # Transformer layers; recommended: 2–12

# === Text Generation Parameters ===
temperature = 0.7         # Controls creativity; lower = more deterministic, higher = more random
top_k = 4                 # Restricts choices to top-k words; lower = more coherent, higher = more diverse

# === Notes ===
# - `num_heads` must evenly divide `embed_size` to avoid shape errors.
# - Increasing `seq_len`, `embed_size`, or `num_layers` requires more memory.
# - High `temperature` or `top_k` values might reduce coherence in generated text.
```

#### Explanation of Configuration Parameters

- **Training Parameters**: Control the length of training (`num_epochs`), real-time output, and whether to ignore cached data.
- **Model Architecture**: These parameters determine the complexity and capacity of the transformer model. For example:
  - `seq_len` controls the number of tokens in each training sample.
  - `embed_size` and `hidden_dim` affect how much information the model retains and processes.
  - `num_heads` and `num_layers` specify the structure of the transformer, affecting how it captures contextual relationships.
- **Text Generation**:
  - `temperature` adjusts how creative the AI is—higher values encourage more varied word choice.
  - `top_k` limits generation to the top-k most likely words, balancing coherence with diversity.

---

## Under the Hood

Let’s dive deeper into the functionality of MiniAI and see what happens behind the scenes.

### Step 1: Reading and Preparing the Data
MiniAI uses **PyMuPDF** to read text from a PDF file. Once the text is extracted, it’s tokenized (split into individual words), lowercased, and punctuations are adjusted to maintain structure. A vocabulary is then built, where each unique word is assigned a unique ID, and three special tokens—`<unk>`, `<pad>`, `<eos>`—are added:
   - **Unknown token** (`<unk>`) handles out-of-vocabulary words.
   - **Padding token** (`<pad>`) ensures sequences are of equal length.
   - **End of Sequence token** (`<eos>`) marks the end of each sequence.

### Step 2: Model Caching
MiniAI can save a trained model to disk, referred to as “caching,” making it possible to reuse the model without retraining. This cache includes:
   - **Model Parameters**: Learned weights and biases of the model.
   - **Vocabulary**: The word-to-ID mapping generated during the preprocessing phase.
   - **Configuration**: Key settings used in training, allowing for seamless reloading.

When MiniAI starts, it checks if a cache exists. If so, it will load the cached model unless `overwrite_cache` is set to `True`.

### Step 3: Training the Transformer Model
Training is where MiniAI learns patterns in language using a transformer model. This transformer architecture is key to capturing relationships across words. Here’s what’s happening in the `LanguageModelTrainer`:
   - **Transformer Model Structure**:
     - Each **transformer layer** has self-attention and feed-forward layers.
     - **Multi-head self-attention** helps the model look at words in context, finding relationships across the sequence.
     - **Feed-forward networks** add flexibility to the model, enabling it to better understand complex sentence structures.
   - **Loss Calculation**: The model calculates a loss by comparing its predicted words to actual words in the training data, iteratively adjusting its parameters to improve accuracy.
   - **Grammar-Aware Tasks**: For better generalization, MiniAI performs:
     - **Masked Language Modeling (MLM)**: Random words in the sequence are masked, and the model must predict them based on the context. This builds contextual awareness.
     - **Sequence Reordering**: Parts of a sentence are scrambled, and the model learns to predict the correct order, enhancing grammatical understanding.

### Step 4: Generating Text
Once trained, MiniAI can generate text based on a prompt. Here’s how it does it:
   - **Input Prompt**: You provide a prompt, such as “Once upon a time,” which is tokenized and fed into the model.
   - **Prediction and Sampling**: The model predicts the next word based on the current sequence, with options adjusted by:
     - **Temperature**: Controls randomness in predictions. A lower temperature focuses on the most probable next word, while a higher temperature introduces variety.
     - **Top-K Sampling**: Restricts predictions to the top-K most probable words, enhancing coherence while allowing some flexibility.
   - **Word-by-Word Generation**: The predicted word is appended to the prompt, and the model iterates, generating the next word until the desired length is reached or an `<eos>` token is generated.

By following this process, MiniAI produces text that mirrors the language style and structure of the original PDF. This approach to language modeling provides a flexible foundation

 for tasks ranging from creative writing to language exploration.

## License
This project is licensed under the MIT License.