import numpy as np
import torch
import torch.nn as nn
from components.transformers import SingleEncoderLayer
from components.transformers import ScaledProductAttentionSoftmax
from components.sinkformers import ScaledProductAttentionSinkhorn


def generate_sinusoid_table(seq_len, d_model):
    """
    Generate a positional encoding table (using sinusoidal table)
    :param seq_len: the max sequence length
    :param d_model: the expected number of features of the input
    :return: the positional encoding table
    """

    def get_angle(pos, i, d_model):
        return pos / np.power(10000, (2 * (i // 2)) / d_model)

    sinusoid_table = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
            else:
                sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

    return torch.FloatTensor(sinusoid_table)


class NLPTransformerEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 sequence_length,
                 pad_id,
                 n_layers=6,
                 d_model=512,
                 n_heads=8,
                 p_dropout=0.1,
                 d_hidden=2048,
                 n_iter=3,
                 mode="softmax"):
        """
        Define the Transformer Encoder layer (contains N encoders) specifically designed for NLP sentiment prediction
        task
        :param vocab_size: the size of the vocabulary (for the embedding layer)
        :param sequence_length: the max length of the input sequence
        :param n_layers: number of encoders
        :param d_model: the features of input
        :param n_heads: number of heads in Multi-head attention
        :param p_dropout: dropout rate of encoder
        :param d_hidden: hidden size of the feed-forward network in the encoder
        :param pad_id: padding id
        """
        super(NLPTransformerEncoder, self).__init__()

        # Cache the variables
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.d_hidden = d_hidden
        self.pad_id = pad_id

        # Create positional_encoding_table
        self.encoding_table = generate_sinusoid_table(
            seq_len=sequence_length + 1,
            d_model=self.d_model
        )

        # Create embedding layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )
        self.positional_embedding = nn.Embedding.from_pretrained(
            self.encoding_table,
            freeze=True
        )

        # Create a series of Encoder
        self.encoders = nn.ModuleList(
            [
                SingleEncoderLayer(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    p_dropout=self.p_dropout,
                    d_hidden=self.d_hidden,
                    attention_class=ScaledProductAttentionSoftmax if mode == "softmax" else ScaledProductAttentionSinkhorn,
                    attention_params={} if mode == "softmax" else {'n_iter': n_iter, 'eps': 1}
                )
                for _ in range(self.n_layers)
            ]
        )

        # Linear to classification layer
        self.linear = nn.Linear(d_model, 2)

        # Softmax layer to calculate confidence score
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        """
        Perform the forward operations of the inputs
        :param inputs: inputs of size (batch_size, sequence_length)
        :return: the outputs of the encoder layer
        """
        # Generate the positional encoding of the input sequences
        # positions has size (batch_size, sequence_length)
        batch_size, sequence_length = inputs.size(0), inputs.size(1)
        positions = (torch.arange(sequence_length, device=inputs.device, dtype=inputs.dtype).repeat(batch_size, 1)
                     + 1)

        # Fill the padding position with 0
        # positions has size (batch_size, sequence_length)
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)

        # Sum the embeddings
        # outputs has shape (batch_size, sequence_length, d_model)
        outputs = self.embedding(inputs) + self.positional_embedding(positions)

        # Create the attention mask
        # attention_masks has size (batch_size, sequence_length, sequence_length)
        attention_masks = (inputs.eq(self.pad_id)
                           .unsqueeze(1)
                           .repeat(1, inputs.size(1), 1))

        # Save the attention weights of each encoder layers
        attention_weights = []

        # Pass through encoder layers
        for encoder in self.encoders:
            # Pass through encoder to get the result
            # outputs has shape (batch_size, sequence_length, d_model)
            # attention_weights has shape (batch_size, n_heads, sequence_length, sequence_length)
            outputs, attention_weight = encoder(outputs, attention_masks)
            # Append to the list of weights
            attention_weights.append(attention_weight)

        # Perform average on the sequence_length dimension
        # outputs has size (batch_size, d_model)
        outputs, _ = torch.max(outputs, dim=1)

        # Pass through linear layer
        # outputs has size (batch_size, 2)
        outputs = self.linear(outputs)
        outputs = self.softmax(outputs)

        # Return the outputs and attention weights
        return outputs, attention_weights
