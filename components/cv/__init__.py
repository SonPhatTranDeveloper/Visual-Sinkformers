"""
This file contains the component implementation of Visual Transformers architecture
"""
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_input, d_hidden, p_dropout):
        """
        Initialize a feed forward network with 2 layers and GELU activation function
        :param d_input: the dimension of the input
        :param d_hidden: the dimension of the hidden layer
        :param p_dropout: dropout rate for the parameters
        """
        super(FeedForwardNetwork, self).__init__()

        # Create network
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=d_input, out_features=d_hidden),
            nn.GELU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features=d_hidden, out_features=d_input),
            nn.Dropout(p=p_dropout),
        )

    def forward(self, inputs):
        """
        Feed forward function of the network
        :param inputs: shape (batch_size, d_1, d_2, ..., d_input)
        :return: outputs: shape (batch_size, d_1, d_2, ..., d_input)
        """
        return self.feed_forward(inputs)


class PreNormalizationNetwork(nn.Module):
    def __init__(self, d_model, func):
        """
        Define the PreNormalizationNetwork for the calculating LayerNorm(Function(X))
        See ViT paper for more details
        :param d_model: the dimension of the input for Layer Normalization
        :param func: the function to apply, which can be Multi-head Self-attention (MSA) or Multi-layered perceptron
        """
        super(PreNormalizationNetwork, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.func = func

    def forward(self, inputs):
        """
        Feed forward function, inputs has shape (batch_size, d_0, d_1, ..., d_model)
        :param inputs: shape (batch_size, d_0, d_1, ..., d_model)
        :return: outputs shape (batch_size, d_0, d_1, ..., d_model)
        """
        return self.func(self.layer_norm(inputs))


class ScaledProductAttentionSoftmax(nn.Module):
    def __init__(self, d_model, n_heads, d_head, p_dropout):
        """
        Initialized the scaled product attention with softmax normalization
        :param d_model: the dimension of input features
        :param n_heads: the number of head of multi-head attention
        :param d_head: the number of features per head
        :param p_dropout: the dropout rate
        """
        # Initialize model
        super(ScaledProductAttentionSoftmax, self).__init__()

        # Cache the variables
        self.d_model = d_model
        self.n_head = n_heads
        self.d_head = d_head
        self.p_dropout = p_dropout

        # Calculate the inner dimension of multi-headed attention
        self.inner_dim = self.n_head * self.d_head

        # If number of head is one and d_model == d_head => We don't project the final output
        self.project_output = False if (self.n_head == 1 and self.d_model == self.d_head) else True

        # Create the Softmax layer for the last dimension
        self.softmax = nn.Softmax(dim=-1)

        # Mapping the input to (query, key, value)
        # no bias
        self.map_to_qkv = nn.Linear(in_features=self.d_model, out_features=self.inner_dim * 3, bias=False)

        # Output projection if any
        # else it is just an identity layer
        self.map_to_output = nn.Sequential(
            nn.Linear(self.inner_dim, self.d_model),
            nn.Dropout(p=self.p_dropout)
        ) if self.project_output else nn.Identity()

    def forward(self, inputs):
        """
        Feed forward of scaled product attention
        :param inputs: shape (batch_size, number_of_patches, d_model) where d_model = n_head * d_head
        :return: outputs: shape (batch_size, number_of_patches, d_model)
        """
        # Mapping to keys, queries and values
        # queries_keys_values is a list of tensor of size (batch_size, number_of_batches, d_model)
        # or (batch_size, number_of_batches, n_head * d_head)
        queries_keys_values = self.map_to_qkv(inputs).chunk(3, dim=-1)

        # Split queries, keys and values
        # b is batch_size, n is number_of_patches, h is number of heads, d is the dimension of each head
        # Each of q, k, v has shape (batch_size, n_heads, number_of_patches, d_head)
        q, k, v = map(lambda item: rearrange(item, 'b n (h d) -> b h n d', h=self.n_head), queries_keys_values)

        # Calculate the attention scores and weights
        # attention_scores has shape (batch_size, n_heads, number_of_patches, number_of_patches)
        # attention_weights has shape (batch_size, n_heads, number_of_patches, number_of_patches)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_head)
        attention_weights = self.softmax(attention_scores)

        # Calculate the output result
        # outputs has size (batch_size, n_heads, number_of_batches, d_head)
        outputs = torch.matmul(attention_weights, v)

        # Reshape outputs to (batch_size, number_of_batches, (n_heads x d_head) = d_model)
        outputs = rearrange(outputs, 'b h n d -> b n (h d)')

        # Return the outputs and attention_weights
        return outputs, attention_weights


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_head, d_ff, p_dropout, attention_func, attention_params):
        """
        Initialize the Transformer Encoder with multiple layers of multi-headed self-attention blocks (MSA)
        :param d_model: the dimension of input features
        :param n_layers: the number of layers of the Transformer encoder
        :param n_heads: the number of head of the Multi-headed attention block
        :param d_head: the dimension of each head in the Multi-headed self-attention block (MSA)
        :param d_ff: the hidden dimension of the feed-forward network (MLP)
        :param p_dropout: the dropout rate
        :param attention_func: the attention type (softmax or sinkhorn)
        """
        # Initialize the layer
        super(TransformerEncoder, self).__init__()

        # Cache the variables
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.p_dropout = p_dropout

        # Create single encoder block
        self.layers = nn.ModuleList([])

        # Parameters for attention class
        params = {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_head': self.d_head,
            'p_dropout': self.p_dropout,
            **attention_params
        }

        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNormalizationNetwork(d_model=d_model, func=attention_func(**params)),
                PreNormalizationNetwork(d_model=d_model, func=FeedForwardNetwork(
                    d_input=d_model,
                    d_hidden=d_ff,
                    p_dropout=p_dropout
                ))
            ]))

    def forward(self, inputs):
        """
        Feed forward function of the multi-layered encoder
        :param inputs: shape (batch_size, number_of_patches, d_model)
        :return: outputs: shape (batch_size, number_of_patches, d_model)
        """
        # Save the attention weights
        weights = []

        # Perform feed forward operation for the layers
        for attention, feed_forward in self.layers:
            # Calculate the attention result
            # attention_result has shape (batch_size, number_of_patches, d_model)
            # attention_weight has shape (batch_size, number_of_heads, number_of_patches, number_of_patches)
            attention_result, attention_weight = attention(inputs)

            # Perform LayerNorm(MSA) + inputs
            inputs = attention_result + inputs

            # Perform LayerNorm(MLP) + inputs
            inputs = feed_forward(inputs) + inputs

            # Append to weights
            weights.append(attention_weight.cpu().detach().numpy())

        # Return result and attention_weights
        # inputs has shape (batch_size, number_of_patches, d_model)
        # attention_weights is a list containing numpy arrays of
        # size (batch_size, number_of_heads, number_of_patches, number_of_patches)
        return inputs, weights


class VisualTransformerClassification(nn.Module):
    def __init__(self,
                 image_width, image_height,
                 patch_width, patch_height,
                 n_classes,
                 d_model,
                 n_layers,
                 n_heads,
                 d_ff,
                 pooling='cls',
                 n_channels=3,
                 d_head=64,
                 p_dropout=0.0,
                 p_emb_dropout=0.0,
                 attention_class='softmax'):
        """
        Initialize the Visual Transformer for classification model
        :param image_width: the width of input image
        :param image_height: the height of input image
        :param patch_width: the width of the patch
        :param patch_height: the height of the patch
        :param n_classes: number of class for classification task
        :param d_model: the size of input features into the Transformer encoder block
        :param n_layers: the number of layer of Transformer encoder block
        :param n_heads: the number of attention head of Transformer encoder block
        :param d_ff: the hidden size of MLP
        :param pooling: pooling type
        :param n_channels: the number of channels of the image
        :param d_head: the number of heads of attention block in Transformer encoder
        :param p_dropout: the dropout rate for the Transformer encoder block
        :param p_emb_dropout: the dropout rate for the positional embedding layer
        :param attention_class: class of the attention mechanism (Softmax or Sinkhorn)
        """
        # Check the configuration of size
        assert image_width % patch_width == 0 and image_height % patch_height == 0, \
            "Image dimensions should be divisible by the patch size"

        # Check the type of pooling
        assert pooling in ['cls', 'mean'], "Pooling must be either cls (classification token) or mean (mean pooling)"

        # Check the type of attention class
        assert attention_class in ['softmax', 'sinkhorn'], "Attention class must either be softmax or sinkhorn"

        # Save the image configuration
        self.image_width, self.image_height = image_width, image_height
        self.patch_width, self.patch_height = patch_width, patch_height
        self.n_channels = n_channels

        # Save number of classes
        self.n_classes = n_classes

        # Save pooling type
        self.pooling = pooling

        # Calculate the number of patches
        number_of_patches = (self.image_width // self.patch_width) * (self.image_height // self.patch_height)
        patch_dims = self.n_channels * self.patch_height * self.patch_width

        # Layer to convert image from patches to embeddings
        # b is the batch size, c is the channels,
        # (h p1) = number_of_patches * patch_height, (2, p2) = number of patches * patch_width
        self.image_to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(in_features=patch_dims, out_features=d_model)
        )

        # Positional encoding layer
        self.positional_embedding = nn.Parameter(torch.randn(1, number_of_patches + 1, d_model))

        # Classification token layer
        self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))

        # Drop-out layer for embedding
        self.drop_out = nn.Dropout(p=p_emb_dropout)

        # Transformer layer
        self.transformer_encoder = None
        if attention_class == "softmax":
            self.transformer_encoder = TransformerEncoder(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_head=d_head,
                d_ff=d_ff,
                p_dropout=p_dropout,
                attention_func=ScaledProductAttentionSoftmax,
                attention_params={}
            )

        # Map-to-output layers
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model),
            nn.Linear(in_features=d_model, out_features=n_classes)
        )

    def forward(self, images):
        """
        Forward operation of the Visual Transformer for classification
        :param images: tensor of size (batch_size, channels, height, width)
        :return: outputs
        """
        # Map to patches
        # patches has shape (batch_size, number of patches, d_model)
        patches = self.image_to_patch_embedding(images)

        # Get the batch size and number of patches
        batch_size, number_of_batches, _ = patches.shape

        # Concatenate the classification token into the patches
        # cls_tokens has shape (batch_size, 1, d_model)
        # patches has shape (batch_size, number_of_patches + 1, d_model)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        patches = torch.cat([cls_tokens, patches], dim=1)

        # Add the positional embedding
        # patches has shape (batch_size, number_of_patches + 1, d_model)
        patches = patches + self.positional_embedding[:, :(number_of_batches + 1)]

        # Go through embedding drop out layer
        # patches has shape (batch_size, number_of_patches + 1, d_model)
        patches = self.drop_out(patches)

        # Go the Transformer layer
        # outputs has size (batch_size, number_of_patches + 1, d_model)
        # attention_weights is a list containing n_layers tensor of shape
        # (batch_size, n_heads, number_of_patches, number_of_patches)
        outputs, attention_weights = self.transformer_encoder(patches)

        # Perform mean pooling or grap the classification token
        if self.pooling == 'mean':
            outputs = outputs.mean(dim=1)
        else:
            outputs = outputs[:, 0]

        # Map to the classification head
        # outputs has size (batch_size, n_classes)
        outputs = self.feed_forward(outputs)
        return outputs, attention_weights

