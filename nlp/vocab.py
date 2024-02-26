"""
Define a class called Vocabulary that is used to convert text into ids
"""
from collections import Counter, OrderedDict
from mosestokenizer import MosesTokenizer
import argparse


class Vocabulary:
    def __init__(self, max_vocab_size: int = 16000,
                 pad_token: str = "<PAD>",
                 unknown_token: str = "<UNK>",
                 start_token: str = "<START>",
                 end_token: str = "<END>"):
        """
        Initialize the Vocabulary object
        :param max_vocab_size: the maximum size of the vocabulary
        :param pad_token: padding token <PAD>
        :param unknown_token: unknown token <UNK>
        :param start_token: start token <START>
        :param end_token: end token <END>
        """
        # Cache the variables
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.unknown_token = unknown_token
        self.start_token = start_token
        self.end_token = end_token

        # Cache the special token
        self.special_token = [self.pad_token, self.unknown_token,
                              self.start_token, self.end_token]

        # Save the frequency of the tokens
        self.frequencies = Counter()

        # Create token to id mapping
        self.token_to_id_mapping = OrderedDict()
        for token in self.special_token:
            self.token_to_id_mapping[token] = len(self.token_to_id_mapping)

    def build(self, corpus_path: str, external_tokenizer, max_sentence_length: int = 10000):
        """
        Build the vocabulary given the path of the corpus, the external tokenizer (SentencePiece or somthing)
        :param corpus_path: File path to the corpus
        :param external_tokenizer: the external tokenizer used to tokenize the text
        :param max_sentence_length: the maximum length of the sentence
        :return: None
        """
        # Open and read the corpus, and count the frequencies of the tokens
        with open(corpus_path, 'r', encoding='utf-8') as reader:
            for i, line in enumerate(reader.readlines()):
                if len(line) >= max_sentence_length:
                    line = line[:max_sentence_length]
                tokens = external_tokenizer(line.strip())
                self.frequencies.update(tokens)

        # Only put the tokens with the highest frequencies to the vocabulary
        for token, freq in self.frequencies.most_common(self.max_vocab_size - len(self.special_token)):
            self.token_to_id_mapping[token] = len(self.token_to_id_mapping)

    def save(self, path: str, postfix: str = ".vocab"):
        """
        Save the vocabulary into a file
        :param path: the path to save the vocab file
        :param postfix: the postfix to save the vocab
        :return: None
        """
        with open(path + postfix, 'w', encoding='utf-8') as writer:
            for token, token_id in self.token_to_id_mapping.items():
                writer.write('{token}\t{id}\n'.format(token=token, id=token_id))


def build(arguments):
    """
    Create and save the vocabulary
    :return: None
    """
    # Create the
    tokenizer = MosesTokenizer('en')

    vocab = Vocabulary(
        max_vocab_size=arguments.vocab_size,
        pad_token=arguments.pad_token,
        unknown_token=arguments.unknown_token,
        start_token=arguments.start_token,
        end_token=arguments.end_token
    )

    # Build and save the vocabulary
    vocab.build(arguments.corpus, tokenizer, arguments.max_sentence_length)
    vocab.save(arguments.prefix)


if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Parse the arguments
    parser.add_argument('--corpus', required=True, type=str, help='../data/imdb_corpus.txt')
    parser.add_argument('--prefix', required=True, type=str, help='output vocab(or sentencepiece model) name prefix')
    parser.add_argument('--vocab_size', default=16000, type=int, help='the maximum size of the vocabulary')
    parser.add_argument('--max_sentence_length', default=100000, type=int, help='The maximum input sequence length')
    parser.add_argument('--pad_token', default='<PAD>', type=str, help='token that indicates padding')
    parser.add_argument('--unknown_token', default='<UNK>', type=str, help='token that indicates unknown word')
    parser.add_argument('--start_token', default='<START>', type=str, help='token that indicates beginning of sentence')
    parser.add_argument('--end_token', default='<END>', type=str, help='token that indicates end of sentence')
    args = parser.parse_args()

    # Build and save the vocabulary
    build(args)
