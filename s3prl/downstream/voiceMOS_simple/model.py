import torch
import torch.nn as nn

from s3prl.downstream.model import AttentivePooling, MeanPooling

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)


    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class Model(nn.Module):
    def __init__(self, input_dim, pooling_name, **kwargs):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.pooling = eval(pooling_name)(input_dim=input_dim, activation='ReLU')

    def forward(self, features, features_len):
        features, _ = self.pooling(features, features_len)
        scores = self.linear(features)

        return scores.squeeze(-1)