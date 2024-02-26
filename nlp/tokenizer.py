"""
This file defines the Tokenizer class, which is responsible for tokenizing a sentence,
as well as create mapping between tokens and ids, as well as ids to tokens
Author: Son Phat Tran
"""
from collections import OrderedDict
from typing import List


class Tokenizer:
    def __init__(self, tokenizer, vocab_file_path: str,
                 pad_token: str = "<PAD>",
                 unknown_token: str = "<UNK>",
                 start_token: str = "<START>",
                 end_token: str = "<END>"):
        """
        Initialize the tokenizer class to tokenizer text
        :param tokenizer: the external tokenizer use (maybe a function)
        :param vocab_file_path: the path containing the vocabulary
        :param pad_token: padding token <PAD>
        :param unknown_token: unknown token <UNK>
        :param start_token: start token <START>
        :param end_token: end token <END>
        """
        # Save the tokenizer and special tokens
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token

        # Load in the vocabulary file
        self.token_to_id_mapping = OrderedDict()
        self.id_to_token_mapping = OrderedDict()

        # Build token to id mapping
        with open(vocab_file_path, 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                token, token_id = line.split()
                self.token_to_id_mapping[token] = int(token_id)

        for token, token_id in self.token_to_id_mapping.items():
            self.id_to_token_mapping[token_id] = token

    def tokenize(self, text: str) -> List[str]:
        """
        Convert the text into a list of tokens
        :param text: the text to convert
        :return: the list of tokens
        """
        return self.tokenizer(text)

    def convert_token_to_id(self, token: str) -> int:
        """
        Convert a token to the id in the vocabulary
        :param token: the token to convert
        :return: id in the vocabulary
        """
        return self.token_to_id_mapping.get(token, self.token_to_id_mapping.get(self.unknown_token))

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to list of ids in the vocab
        :param tokens: the list of tokens to convert
        :return: list of ids
        """
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, token_id: int) -> str:
        """
        Convert an id (integer) in a token (str) using the vocab.
        """
        return self.id_to_token_mapping.get(token_id, self.unknown_token)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of ids in list of tokens using the vocab.
        """
        return [self.convert_id_to_token(token_id) for token_id in ids]

    @property
    def vocab_size(self) -> int:
        """
        Get the vocabulary size.
        """
        return len(self.token_to_id_mapping)

    @property
    def pad_token_id(self) -> int:
        """
        Get the id of pad_token in the vocab.
        """
        return self.convert_token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """
        Get the id of unknown_token in the vocab.
        """
        return self.convert_token_to_id(self.unknown_token)

    @property
    def bos_token_id(self) -> int:
        """
        Get the id of start_token in the vocab.
        """
        return self.convert_token_to_id(self.start_token)

    @property
    def eos_token_id(self) -> int:
        """
        Get the id of end_token in the vocab.
        """
        return self.convert_token_to_id(self.end_token)

