import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.gpo import GPO
from models.modules.mlp import MLP
from models.modules.resnet import ResnetFeatureExtractor


class ImageEncoder(nn.Module):
    def __init__(
        self,
        encoder_type='region',
        pooling='gpo-selfattn',
        dropout=0.1,
        embed_dim=1024,
        img_pos_embed_dim=16,
        grid_dropout=0.2, # for grid features only
    ):
        """Initializes image encoder.

        Args:
        * encoder_type: 'region' or 'grid', what kind of image feature to use.
        * pooling: 'avg' or 'gpo-self-attn', what kind of image pooling method to use.
        * dropout: dropout rate for the input image feature.
        * embed_dim: dimension of the embedding space.
        * img_pos_embed_dim: dimension of the positional embedding to be added to the image features.
        * grid_dropout: (only for grid features), dropout rate to set feature vectors in the image grid feature to 0.
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.pooling = pooling

        if encoder_type == 'grid':
            self.backbone_cnn = ResnetFeatureExtractor('wsl', weights_path='', fixed_blocks=0)

        if pooling == 'avg':
            # average pooling
            self.fc = nn.Linear(2048, embed_dim)
        elif pooling == 'gpo-selfattn':
            # Self-attention followed by GPO
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(2048, embed_dim)
            if encoder_type == 'region':
                self.mlp = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(True),
                    nn.Linear(512, embed_dim)
                )

            # self attention layer to contextualize region features
            nheads = 16
            self.lnorm = nn.LayerNorm(embed_dim)
            self.lnorm2 = nn.LayerNorm(embed_dim)
            self.fc = nn.Linear(embed_dim, embed_dim)
            self.fc2 = nn.Linear(embed_dim, embed_dim)
            self.pos_linear = nn.Linear(img_pos_embed_dim, embed_dim) # linear layer for position embedding
            self.dropout_pos = nn.Dropout(dropout) # dropout layer for positional embedding
            self.attn_pool = nn.MultiheadAttention(embed_dim, nheads, dropout=0.0, batch_first=True)

            self.gpool = GPO(32, 32)

        # for grid features only
        self.grid_dropout = grid_dropout

    def forward(self, batch):
        if self.encoder_type == 'region':
            return self.forward_region(batch)
        else:
            return self.forward_grid(batch)
    
    def forward_region(self, batch):
        img = batch['img'] # (bs, n_regions, d)
        bs, n_regions = img.shape[:2]

        # actual number of regions after dropout
        img_length = batch['img_length'] if 'img_length' in batch else None # (bs)

        # padding mask to show which are padded regions
        img_padding_mask = batch['img_padding_mask'] if 'img_padding_mask' in batch else None

        if self.pooling == 'avg':
            img = self.fc(img)
            img = torch.mean(img, dim=1) # (bs, 2048)
        elif self.pooling == 'gpo-selfattn':
            img = self.dropout(img)
            feat = self.linear(img)
            if self.encoder_type == 'region':
                feat = self.mlp(img.view(bs*n_regions, -1)).view(bs, n_regions, -1) + feat

            # add positional embedding to each region
            pos_embeds = batch['img_pos_embeds']
            pos_embeds = self.pos_linear(pos_embeds) # (bs, n_regions, d)
            pos_embeds = self.dropout_pos(pos_embeds)
            res = self.lnorm(feat + pos_embeds)

            # apply self attention
            res, _ = self.attn_pool(query=res, key=res, value=res, key_padding_mask=img_padding_mask)
            feat = feat + res
            res = self.lnorm2(feat)
            res = self.fc2(F.relu(self.fc(res)))
            feat = feat + res

            # apply GPO
            assert img_length is not None
            feat, pool_weights = self.gpool(feat, img_length)
            img = feat

        return img

    def forward_grid(self, batch):
        img = batch['img']
        base_features = self.backbone_cnn(img)

        if self.training:
            # Size Augmentation during training, randomly drop grids (based on GPO paper)
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > self.grid_dropout:
                    feat_i = base_features[i][np.where(rand_list_1[i] > self.grid_dropout * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        batch['img'] = base_features
        batch['img_length'] = feat_lengths

        return self.forward_region(batch)


def build_image_encoder(cfg):
    image_encoder = ImageEncoder(
        encoder_type=cfg.MODEL.img_encoder_type,
        pooling=cfg.MODEL.img_pooling_type,
        dropout=cfg.MODEL.img_dropout,
        embed_dim=cfg.MODEL.embed_dim,
        img_pos_embed_dim=cfg.MODEL.img_pos_embed_dim,
        grid_dropout=cfg.TRAIN.grid_dropout
    )
    return image_encoder
