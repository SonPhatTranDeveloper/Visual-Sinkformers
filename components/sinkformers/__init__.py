"""
This file contains the implementation of the Sinkhorn component of Transformers
"""
import torch
import torch.nn as nn
import numpy as np


class SinkhornDistanceFast(nn.Module):
    def __init__(self, eps, max_iter):
        """
        Initialize the Sinkhorn Fast and Stable AutoDiff algorithm
        Adapted from the paper: https://arxiv.org/pdf/1607.05816.pdf
        :param eps: the epsilon in the Sink
        :param max_iter: the number of Sinkhorn iteration
        """
        # Initialize
        super(SinkhornDistanceFast, self).__init__()

        # Cache the parameters
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, attention_score):
        """
        Perform the Sinkhorn algorithm on the attention score matrix
        :param attention_score: attention_score matrix of shape (batch_size, number_of_query, number_of_value)
        :return: the processed attention_score matrix after the Sinkhorn algorithm
        """
        # Create the cost matrix, which is the negative of the attention score
        cost = - attention_score

        # Get the dimensions
        x_points = cost.shape[-2]
        y_points = cost.shape[-1]
        batch_size = cost.shape[0]

        # Create two marginals mu and nu with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=cost.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False, device=cost.device).fill_(1.0 / y_points).squeeze()

        # Create two vectors u and v with same dimension as mu and nu
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Threshold to stop Sinkhorn
        threshold = 1e-12
        err = None

        # Perform Sinkhorn iterations
        for i in range(self.max_iter):
            if i % 2 == 0:
                u1 = u  # useful to check the update
                u = self.eps * (torch.log(mu) - torch.logsumexp(self.modified_cost(cost, u, v), dim=-1)) + u
                err = (u - u1).abs().sum(-1).mean()
            else:
                v = self.eps * (torch.log(nu) - torch.logsumexp(self.modified_cost(cost, u, v).transpose(-2, -1), dim=-1)) + v
                v = v.detach().requires_grad_(False)
                v[v > 9 * 1e8] = 0.0
                v = v.detach().requires_grad_(True)

            if err.item() < threshold:
                break

        # Calculate the result pi
        pi = torch.exp(self.modified_cost(cost, u, v))
        return pi

    def modified_cost(self, cost_matrix, u, v):
        """
        Calculate the modified cost for logarithmic updates
        :param cost_matrix:
        :param u:
        :param v:
        :return:
        """
        "Modified cost for logarithmic updates"
        return (-cost_matrix + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


class ScaledProductAttentionSinkhorn(nn.Module):
    def __init__(self, d_q, n_iter):
        """
        Initialize the scaled product attention block WITH SINKHORN with query size of d_q
        and n_iter of Sinkhorn iteration
        """
        super(ScaledProductAttentionSinkhorn, self).__init__()
        self.d_q = d_q
        self.n_iter = n_iter

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
        attention_score_shape = attention_score.shape

        # Perform softmax to calculate the weight of the attention score
        # attention_weights has size (batch_size, number_of_heads, number_of_query, number_of_key)
        # attention_score has shape (batch_size x number_of_heads, number_of_query, number_of_key)
        attention_score = attention_score.view(-1, attention_score_shape[2], attention_score_shape[3])

        # Perform Sinkhorn iteration to calculate attention weights
        sinkhorn = SinkhornDistanceFast(eps=1, max_iter=self.n_iter)
        attention_weights = sinkhorn(attention_score)
        attention_weights = attention_weights * attention_weights.shape[-1]
        attention_weights = attention_weights.view(attention_score_shape)

        # Calculate the output
        # output has size (batch_size, number_of_heads, number_of_query, value_size)
        outputs = torch.matmul(attention_weights, v)

        # Return the output and attention weights
        return outputs, attention_weights

