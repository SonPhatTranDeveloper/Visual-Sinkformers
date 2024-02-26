"""
Define the Transformer encoder block and its various components
Author: Son Phat Tran
"""
import numpy as np
import torch
import torch.nn as nn


class ScaledProductAttention(nn.Module):
    def __init__(self, d_q):
        """
        Initialize the scaled product attention block with query size of d_q
        """
        super(ScaledProductAttention, self).__init__()
        self.d_q = d_q

    def forward(self, q, k, v, attention_mask):
        """
        Perform forward operation
        :param q: the query of size (batch_size, number_of_heads, number_of_query, query_size)
        :param k: the key of size (batch_size, number_of_heads, number_of_key, query_size)
        :param v: the value of size (batch_size, number_of_heads, number_of_key, value_size)
        :param attention_mask: attention mask of (batch_size, number_of_heads, number_of_query, number_of_key)
        :return: the output and attention weights
        """
        # Calculate the attention score
        # attention_score has size (batch_size, number_of_heads, number_of_query, number_of_key)
        attention_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_q)

        # Fill the attention score with the mask
        # with an arbitrarily large negative value
        attention_score.masked_fill_(attention_mask, -1e9)

        # Perform softmax to calculate the weight of the attention score
        # attention_weights has size (batch_size, number_of_heads, number_of_query, number_of_key)
        attention_weights = nn.Softmax(dim=-1)(attention_score)

        # Calculate the output
        # output has size (batch_size, number_of_heads, number_of_query, value_size)
        outputs = torch.matmul(attention_weights, v)

        # Return the output and attention weights
        return outputs, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_class):
        """
        Initialize the multi-head attention block
        :param d_model: the number of features in the input
        :param n_heads: the number of heads of the multi-head attention block
        """
        super(MultiHeadAttention, self).__init__()

        # Cache the model size and number of heads
        self.d_model = d_model
        self.n_heads = n_heads

        # Define the size of the key and value
        self.d_k = self.d_v = d_model // n_heads

        # Define the Query, Key, and Value transformation matrix
        self.q_transform = nn.Linear(d_model, d_model)
        self.k_transform = nn.Linear(d_model, d_model)
        self.v_transform = nn.Linear(d_model, d_model)

        # Define the scaled product attention layer
        self.scaled_product_attention = attention_class(self.d_k)

        # Define the linear transformation from the attention heads (n_heads * d_v) to d_model
        self.linear = nn.Linear(n_heads * self.d_v, d_model)

    def forward(self, q, k, v, attention_masks):
        """
        Perform the forward operation
        :param q: the query of size (batch_size, number_of_query, query_size: d_k)
        :param k: the key of size (batch_size, number_of_key, query_size: d_k)
        :param v: the value of size (batch_size, number_of_value, key_size: d_v)
        :param attention_masks: attention mask of size (batch_size, number_of_query, number_of_key)
        :return: the output of multi-head attention and attention_weights
        """
        # Get the batch_size
        batch_size = q.size(0)

        # Perform transformation using the matrices defined above
        # q_heads has size (batch_size, number_of_heads, number_of_query, d_key)
        # k_heads has size (batch_size, number_of_heads, number_of_key, d_key)
        # v_heads has size (batch_size, number_of_heads, number_of_value, d_value)
        q_heads = self.q_transform(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.k_transform(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.v_transform(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Repeat and reshape the attention mask to have
        # size (batch_size, number_of_heads, number_of_query, number_of_key)
        attention_masks = attention_masks.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # Perform attention to get the output and the attention weights
        # attention has size (batch_size, number_of_heads, number_of_query, d_v)
        # attention_weights has size (batch_size, number_of_heads, number_of_query, d_k)
        attention, attention_weights = self.scaled_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attention_masks
        )

        # Reshape attention to get (batch_size, number_of_query, number_of_heads * d_v)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)

        # Linear mapping to get the outputs
        # outputs has size (batch_size, number_of_query, d_model)
        outputs = self.linear(attention)
        return outputs, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_hidden):
        """
        Initialize the feed forward network in the encoder
        :param d_model: the expected number of features of the input
        :param d_hidden: the hidden layer size of the feed forward network
        """
        super(FeedForwardNetwork, self).__init__()

        # Save the model size and hidden size
        self.d_model = d_model
        self.d_hidden = d_hidden

        # Create two linear layers for the feed forward network
        self.linear_1 = nn.Linear(self.d_model, self.d_hidden)
        self.linear_2 = nn.Linear(self.d_hidden, self.d_model)

        # Create a ReLU layer
        self.relu = nn.ReLU()

    def forward(self, inputs):
        """
        Perform forward operation for the inputs
        :param inputs: inputs has shape (batch_size, number_of_inputs, d_model)
        :return: outputs of size (batch_size, number_of_inputs, d_model)
        """
        # Pass through first layer
        # outputs currently has size (batch_size, number_of_inputs, d_hidden)
        outputs = self.relu(self.linear_1(inputs))

        # Pass through the second layer
        # outputs now has size (batch_size, number_of_inputs, d_model)
        outputs = self.linear_2(outputs)

        # Return the outputs
        return outputs


class SingleEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_dropout, d_hidden, attention_class):
        """
        Initialize the encoder block of the Transformers
        :param d_model: the expected number of features of the input (embeddings)
        :param n_heads: the number of scaled-product attention head
        :param p_dropout: the dropout probability
        :param d_hidden: the size of the hidden unit in the feed forward network
        """
        super(SingleEncoderLayer, self).__init__()

        # Cache the dimensions
        self.d_model = d_model
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.d_hidden = d_hidden

        # Create the Multi-head attention layer
        self.multi_head_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            attention_class=attention_class
        )

        # Create the first dropout layer after the multi-headed attention
        self.drop_out_1 = nn.Dropout(p=p_dropout)

        # Create the layer normalization layer
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)

        # Create the feed forward layer
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_hidden=d_hidden
        )

        # Create the second drop-out layer after feed forward
        self.drop_out_2 = nn.Dropout(p=p_dropout)

        # Create the second layer normalization layer
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attention_masks):
        """
        Perform the forward operations of the Transformer Encoder Layer
        :param inputs: the inputs of shape (batch_size, number_of_inputs, d_model)
        :param attention_masks: the attention mask of shape (batch_size, number_of_inputs, number_of_inputs)
        :return: the outputs of encoder block and its attention weights
        """
        # Perform multi-head self-attention
        # attention has shape (batch_size, number_of_inputs, d_model)
        # attention_weights has shape (batch_size, number_of_heads, number_of_inputs, number_of_inputs)
        attention, attention_weights = self.multi_head_attention(
            q=inputs,
            k=inputs,
            v=inputs,
            attention_masks=attention_masks
        )

        # Pass through the drop-out layer
        # attention has shape (batch_size, number_of_inputs, d_model)
        attention = self.drop_out_1(attention)

        # Add + Layer normalization
        # attention has shape (batch_size, number_of_inputs, d_model)
        attention = self.layer_norm_1(inputs + attention)

        # Pass through the feed forward network
        # feed_forward has shape (batch_size, number_of_inputs, d_model)
        feed_forward = self.feed_forward(attention)

        # Perform dropout
        # feed_forward has shape (batch_size, number_of_inputs, d_model)
        feed_forward = self.drop_out_2(feed_forward)

        # Add + Layer normalization
        # feed_forward has shape (batch_size, number_of_inputs, d_model)
        feed_forward = self.layer_norm_2(attention + feed_forward)
        return feed_forward, attention_weights


