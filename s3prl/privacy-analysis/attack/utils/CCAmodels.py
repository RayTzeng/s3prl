import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools
import numpy as np

from cca_zoo.deepmodels import objectives
from cca_zoo.deepmodels.architectures import Encoder, Decoder
from cca_zoo.models import MCCA
from cca_zoo.deepmodels import DCCA, DCCAE



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        # simply take mean operator / no additional parameters
    def forward(self, feature):
        return feature


class DCCAModel(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dims):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        """
        super(DCCAModel, self).__init__()
        
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dims = latent_dims

        encoder_1 = Encoder(latent_dims=self.latent_dims, feature_size=self.x_dim)
        # encoder_2 = Encoder(latent_dims=self.latent_dims, feature_size=self.y_dim)
        encoder_2 = Encoder(latent_dims=self.latent_dims, feature_size=self.y_dim)

        self.model = DCCA(latent_dims=self.latent_dims, encoders=[encoder_1, encoder_2], eps=1e-6)

    def forward(self, x, y):
        return self.model(x, y)

    def loss(self, x, y):
        return self.model.loss(x, y)

    def post_transform(self, z_x, z_y, train=False):
        return self.model.post_transform(z_x, z_y, train=train)

class DCCAEModel(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dims):
        """
        Constructor class for DCCA

        :param latent_dims: # latent dimensions
        :param objective: # CCA objective: normal tracenorm CCA by default
        :param encoders: list of encoder networks
        :param r: regularisation parameter of tracenorm CCA like ridge CCA. Needs to be VERY SMALL. If you get errors make this smaller
        :param eps: epsilon used throughout. Needs to be VERY SMALL. If you get errors make this smaller
        """
        super(DCCAEModel, self).__init__()
        
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dims = latent_dims

        encoder_1 = Encoder(latent_dims=self.latent_dims, feature_size=self.x_dim)
        # encoder_2 = Encoder(latent_dims=self.latent_dims, feature_size=self.y_dim)
        encoder_2 = Encoder(latent_dims=self.latent_dims, feature_size=self.y_dim)
        decoder_1 = Decoder(latent_dims=self.latent_dims, feature_size=self.x_dim)
        decoder_2 = Decoder(latent_dims=self.latent_dims, feature_size=self.y_dim)

        self.model = DCCAE(latent_dims=self.latent_dims, encoders=[encoder_1, encoder_2], decoders=[decoder_1, decoder_2], eps=1e-3)

    def forward(self, x, y):
        return self.model(x, y)

    def loss(self, x, y):
        return self.model.loss(x, y)

    def post_transform(self, z_x, z_y, train=False):
        return self.model.post_transform(z_x, z_y, train=train)

    def score(self, x, y):
        pair_corrs = self.correlations(x, y)
        # n views
        n_views = pair_corrs.shape[0]
        # sum all the pairwise correlations for each dimension. Subtract the self correlations. Divide by the number of views. Gives average correlation
        dim_corrs = (pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views) / (
                n_views ** 2 - n_views)

        return dim_corrs

    def correlations(self, x, y):
        """
        :return: numpy array containing correlations between each pair of views for each dimension (#views*#views*#latent_dimensions)
        """
        transformed_views = self.transform(x, y)
        all_corrs = []
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(np.diag(np.corrcoef(x.T, y.T)[:x.shape[1], y.shape[1]:]))
        all_corrs = np.array(all_corrs).reshape(
            (len(transformed_views), len(transformed_views), -1))
        return all_corrs

    def transform(self, x, y):
        z_x, z_y = self.model(x, y)
        z_list = [z_x.detach().cpu().numpy(), z_y.detach().cpu().numpy()]
        z_list = self.post_transform(*z_list)
        return z_list
