"""
This file defines the functions for creating a movie review dataset
"""
import torch
from torch.utils.data import TensorDataset

from mosestokenizer import MosesTokenizer
from datasets import IMDBDataset
from tokenizer import Tokenizer


def create_dataset(sentiment_analysis_dataset,
                   tokenizer,
                   max_sequence_length=512,
                   device="cpu"):
    """
    Create a Torch Tensor dataset object from the imdb dataset and tokenizer
    :param sentiment_analysis_dataset: the IMDB sentiment dataset object
    :param tokenizer: tokenizer
    :param max_sequence_length: maximum sequence length
    :param device: the device to hold the tensor
    :return: TensorDataset
    """

    # Create a list to save the id tensor
    ids = []

    # Create a list to save the label tensor
    labels = []

    # Append to the tensor
    for text, label in sentiment_analysis_dataset:
        # Tokenize the text
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:max_sequence_length]

        # Convert the tokens to the input id
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Add padding to tokens
        padding_length = max_sequence_length - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)

        # Append to ids and labels
        ids.append(input_ids)
        labels.append(label)

    ids = torch.tensor(ids, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    return TensorDataset(ids, labels)


if __name__ == "__main__":
    # Create dataset
    imdb = IMDBDataset("../data/imdb_dataset.csv")
    tokenizer = Tokenizer(
        tokenizer=MosesTokenizer(),
        vocab_file_path="imdb.vocab"
    )
    dataset = create_dataset(
        imdb,
        tokenizer
    )

    # Get the input_ids and label of the first training examples
    input_ids, label = dataset[0]
