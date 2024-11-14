from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from transformers import BertTokenizer, AutoTokenizer
import os
import gc

# Function to create and train a BertWordPieceTokenizer
def create_and_train_tokenizer(directory, trained_tokenizer_directory, vocab_size=30000, lowercase=True):
    """
    Creates and trains a BertWordPieceTokenizer using text files from the specified directory.

    Args:
        directory (str): The path to the directory containing .txt files for training.
        vocab_size (int): The vocabulary size for the tokenizer.
        lowercase (bool): Whether to convert text to lowercase before tokenization.

    Returns:
        str: The path to the saved tokenizer vocabulary file.
    """

    def text_iterator(directory):
        """
        Generator to yield chunks of text from .txt files in the directory.

        Args:
            directory (str): The path to the directory containing .txt files.

        Yields:
            str: A chunk of text from the files.
        """
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
        for file_path in files:
            gc.collect()  # Free up memory
            with open(file_path, 'r', encoding='utf-8') as file:
                while True:
                    chunk = file.read(100000)  # Read in chunks of 100,000 characters
                    if not chunk:
                        break
                    yield chunk

    # Initialize and train the tokenizer
    tokenizer = BertWordPieceTokenizer(lowercase=lowercase)
    tokenizer.train_from_iterator(text_iterator(directory), vocab_size=vocab_size)

    # Save the tokenizer
    output_dir = trained_tokenizer_directory
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_model(output_dir)

    vocab_file = os.path.join(output_dir, "vocab.txt")
    return vocab_file

# Function to update a tokenizer by replacing unused tokens or adding new tokens
def update_tokenizer(base_directory, vocab_file, num_tokens=9000, mode="add_only", prioitize_scibert=False):
    """
    Updates a tokenizer by either replacing unused tokens or adding new tokens.

    Args:
        base_directory (str): Directory where the modified tokenizer will be saved.
        vocab_file (str): Path to the custom vocabulary file.
        num_tokens (int): Number of new tokens to add or replace.
        mode (str): "replace_unused" or "add_only". Determines how tokens are updated.

    Returns:
        None: Saves the updated tokenizer in the specified directory.
    """

    # Load the base and custom tokenizers
    base_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    custom_tokenizer = BertTokenizer(vocab_file=vocab_file)

    # Get vocabularies as lists of tokens
    tokens_base = list(base_tokenizer.get_vocab().keys())
    tokens_ours = list(custom_tokenizer.get_vocab().keys())

    # Identify unused tokens in the base tokenizer
    unused_tokens = [token for token in tokens_base if token.startswith('[unused')]
    unique_tokens = [token for token in tokens_ours if token not in tokens_base]
    if prioitize_scibert:
        scibert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        tokens_scibert = set(scibert_tokenizer.get_vocab().keys())  # Convert to a set for faster lookup
        tokens_in_scibert = [token for token in unique_tokens if token in tokens_scibert]
        tokens_not_in_scibert = [token for token in unique_tokens if token not in tokens_scibert]
        unique_tokens = tokens_in_scibert + tokens_not_in_scibert

    if mode == "replace_unused":
        # Replace unused tokens with new tokens
        print(f"Starting token replacement for {min(num_tokens, len(unused_tokens))} tokens.")
        for i, new_token in enumerate(unique_tokens[:min(num_tokens, len(unused_tokens))]):
            unused_token = unused_tokens[i]
            base_tokenizer.vocab[new_token] = base_tokenizer.vocab.pop(unused_token)
        print(f"Replaced {min(num_tokens, len(unused_tokens))} unused tokens.")

        # Adjust number of tokens to add if some were already replaced
        num_tokens -= min(num_tokens, len(unused_tokens))
        unique_tokens = unique_tokens[min(len(unused_tokens), len(unique_tokens)):]

    if num_tokens > 0:
        # Add the remaining tokens
        print(f"Adding {num_tokens} tokens to the tokenizer.")
        base_tokenizer.add_tokens(unique_tokens[:num_tokens])

    # Save the updated tokenizer
    os.makedirs(base_directory, exist_ok=True)
    base_tokenizer.save_pretrained(base_directory)
    print(f"Tokenizer updated and saved to {base_directory}.")

# Function to train and update the tokenizer, and save the modified tokenizer
def train_and_update_tokenizer(data_directory, trained_tokenizer_directory, final_directory, vocab_size=30000, num_tokens=9000, mode="add_only",prioritize_scibert=False):
    """
    Combines training a tokenizer and updating a base tokenizer.

    Args:
        data_directory (str): The path to the directory containing .txt files for training.
        base_directory (str): Directory where the modified tokenizer will be saved.
        vocab_size (int): The vocabulary size for the tokenizer.
        num_tokens (int): Number of new tokens to add or replace.
        mode (str): "replace_unused" or "add_only". Determines how tokens are updated.

    Returns:
        None: Saves the modified tokenizer.
    """
    # Step 1: Train the tokenizer and get the vocab file path
    vocab_file = create_and_train_tokenizer(data_directory,trained_tokenizer_directory, vocab_size)

    # Step 2: Update the base tokenizer with the trained vocabulary
    update_tokenizer(final_directory, vocab_file, num_tokens, mode,prioritize_scibert)

# Example usage
if __name__ == "__main__":
    data_directory = "../data/"
    trained_tokenizer_directory='trained_bert'
    final_directory = "modified_bert"
    train_and_update_tokenizer(data_directory, trained_tokenizer_directory,final_directory, vocab_size=30000, num_tokens=9000, mode="replace_unused",prioritize_scibert=True)
