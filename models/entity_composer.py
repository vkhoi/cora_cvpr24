import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pack_padded_sequence

# from models.modules.bert_prefix_model import BertPrefixModel
from models.modules.gpo import GPO
from models.modules.graph_attention import GraphAttention


class EntityComposer(nn.Module):
    def __init__(
        self,
        vocab_size=-1,
        word_embed_dim=300,
        hidden_dim=1024,
        embed_dim=1024,
        composer_n_layers=1,
        composer_method='gru',
        bert_pooling='cls',
        dropout=0.0,
        pretrained_word_embedding=None,
    ):
        """Initializes entity composer class (compose objects and attributes to create entities).

        Args:
        * vocab_size: if not using BERT, this is the size of the vocabulary parsed from the training set. Otherwise, -1.
        * word_embed_dim: if not using BERT, this is the word embedding dimension.
        * hidden_dim: intermediate hidden dimension of the graph attention network.
        * embed_dim: dimension of the embedding space.
        * composer_n_layers: number of layers in the graph attention network.
        * composer_method: what kind of text model to embed the object and attribute nodes.
        * bert_pooling: if using BERT, what kind of pooling method to apply on the output text embeddings.
        * dropout: dropout rate to use in graph attention network.
        * pretrained_word_embedding: numpy array of the pretrained word embedding matrix (such as GloVe).
        """
        super().__init__()
        if vocab_size == -1:
            vocab_size = 30522
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        if composer_method not in ['bert', 'bert-prefix']:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_dim)
            if pretrained_word_embedding is not None:
                self.word_embeddings.weight.data.copy_(pretrained_word_embedding)

        self.composer_method = composer_method
        if composer_method == 'gru':
            self.rnn_ent = nn.GRU(word_embed_dim, word_embed_dim, num_layers=1, batch_first=True, bidirectional=False)
            self.rnn_attr = nn.GRU(word_embed_dim, word_embed_dim, num_layers=1, batch_first=True, bidirectional=False)
        elif composer_method == 'bert':
            self.bert_embedding = BertModel.from_pretrained('bert-base-uncased')
            self.bert_pooling = bert_pooling
        elif composer_method == 'bert-prefix':
            self.bert_prefix_model = BertPrefixModel()
            self.bert_pooling = bert_pooling

        # layer to transform from word embedding to hidden dimension
        self.word_fc = nn.Sequential(
            nn.Linear(word_embed_dim, hidden_dim),
            nn.ReLU(),
        )
        self.word_fc.append(nn.Dropout(dropout))

        # graph attention network to compose object and attribute
        self.composer_n_layers = composer_n_layers
        self.drop_output_ent = nn.Dropout(dropout)
        self.drop_output_attr = nn.Dropout(dropout)
        self.graph_attn = GraphAttention(
            n_layers=composer_n_layers, d=hidden_dim,
        )
        if hidden_dim != embed_dim:
            self.fc_out = nn.Linear(hidden_dim, embed_dim)
    
    def forward(self, batch):
        # object text tokens
        entity_tokens = batch['entity_tokens']
        entity_lengths = batch['entity_lengths']
        entity_padding_mask = batch['entity_padding_mask']

        # attribute text tokens
        entity_attribute_tokens = batch['entity_attribute_tokens']
        entity_attribute_padding_mask = batch['entity_attribute_padding_mask']
        entity_attribute_lengths = batch['entity_attribute_lengths']

        # find which object is really there and not padded objects
        entity_valid_mask = (~entity_padding_mask).sum(-1) > 0 # (bs, max_n_entities)
        batch['entity_valid_mask'] = entity_valid_mask

        # find which attribute is really there and not padded attributes
        entity_attribute_valid_mask = (~entity_attribute_padding_mask).sum(-1) > 0 # (bs, max_n_entities, max_n_attr)

        device = entity_tokens.device
        bs, max_n_entities, max_n_entity_tokens = entity_tokens.shape
        max_n_attributes = entity_attribute_tokens.shape[2]

        # use text model to embed objects from their text tokens
        entity_embeds = self.embed_text_phrase(
            tokens=entity_tokens,
            lengths=entity_lengths,
            token_type='entity'
        ) # (bs, max_n_ent, d)

        # use text model to embed attributes from their text tokens
        entity_attribute_embeds = self.embed_text_phrase(
            tokens=entity_attribute_tokens,
            lengths=entity_attribute_lengths,
            token_type='attribute'
        ) # (bs, max_n_ent, max_n_attr, d)

        # Graph attention to jointly embed entity and attribute. Here, we only care about each object and their
        # corresponding attributes that connect to it. The object nodes are not connected to each other.

        # 1) create adjacency matrix such that:
        # adj_matrix[i] is the graph for object i with (max_n_attrs + 1) nodes.
        # adj_matrix[i] is of size (max_n_attrs + 1, max_n_attrs + 1).
        # Node 0 in adj_matrix[i] is the object node, while nodes 1 -> max_n_attr are the attribute nodes. So, if 
        # object i has 3 attributes, then adj_matrix[i][0, 1:3] = 1 while the rest are 0. In addition, an object node
        # should have self-loop and connect to itself in case where an object does not have any attributes, so
        # adj_matrix[i][0, 0] = 1.
        adj_matrix = torch.cat(
            (
                entity_attribute_valid_mask.view(bs * max_n_entities, max_n_attributes).unsqueeze(-1).float(),
                torch.zeros((bs * max_n_entities, max_n_attributes, max_n_attributes-1), device=device).float(),
            ),
            dim=-1
        ) # (bs * max_n_entities, max_n_attr, max_n_attr)
        adj_matrix = torch.cat(
            (
                torch.zeros((bs * max_n_entities, 1, max_n_attributes), device=device),
                adj_matrix
            ),
            dim=1
        ) # (bs * max_n_ent, max_n_attr+1, max_n_attr)
        adj_matrix = torch.cat(
            (
                adj_matrix,
                torch.zeros((bs * max_n_entities, max_n_attributes+1, 1), device=device)
            ),
            dim=2
        ) # (bs * max_n_ent, max_n_attr+1, max_n_attr+1)
        
        # add self-loop: object node should connect to itself for cases where an object doesn't have any attributes
        adj_matrix[:,0,0] = 1.0

        # symmetry: if object node is connected to attribute node, then attribute node is also connected to object node
        adj_matrix = adj_matrix + adj_matrix.permute(0, 2, 1)
        adj_matrix[adj_matrix > 1e-5] = 1.0

        # 2) run graph attention network
        entity_embeds = entity_embeds.view(bs * max_n_entities, 1, -1)
        entity_attribute_embeds = entity_attribute_embeds.view(bs * max_n_entities, max_n_attributes, -1)

        for t in range(self.composer_n_layers):
            node_embeds = torch.cat((entity_embeds, entity_attribute_embeds), dim=1) # (bs * max_n_ent, max_n_attr+1, d)
            node_embeds = self.graph_attn(node_embeds, adj_matrix, t=t)

            entity_embeds = node_embeds[:,0,:] # node 0 is object node
            entity_attribute_embeds = node_embeds[:,1:,:] # node 1->max_attr+1 are attribute nodes

            if t < self.composer_n_layers - 1:
                # perform dropout on output embeddings
                entity_embeds = self.drop_output_ent(entity_embeds)
                entity_attribute_embeds = self.drop_output_attr(entity_attribute_embeds)

        if self.hidden_dim != self.embed_dim:
            entity_embeds = self.fc_out(entity_embeds)

        entity_embeds = entity_embeds.view(bs, max_n_entities, -1)

        return entity_embeds

    def embed_text_phrase(self, tokens, lengths=None, token_type='entity'):
        if self.composer_method == 'avg':
            # average pool all text tokens
            embeds = self.word_embeddings(tokens)
            embeds = self.word_fc(embeds) # (*, len, d)
            can_avg_mask = (tokens > 2)
            embeds = (embeds * can_avg_mask.unsqueeze(-1)).sum(-2) / (can_avg_mask.sum(-1, keepdim=True) + 1e-10)

        elif self.composer_method == 'gru':
            assert lengths is not None

            embeds = self.word_embeddings(tokens)
            output_shape = list(embeds.shape)
            output_shape = torch.Size(output_shape[:-2] + [output_shape[-1]])
            embeds = torch.flatten(embeds, start_dim=0, end_dim=-3) # (bs', len, d)
            lengths = torch.flatten(lengths)
            lengths = lengths.clamp(min=1)

            packed = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=False)
            if token_type == 'entity':
                _, embeds = self.rnn_ent(packed)
            else:
                _, embeds = self.rnn_attr(packed)

            if embeds.shape[0] > 1:
                embeds = embeds.mean(dim=0)

            embeds = embeds.view(*output_shape)
            embeds = self.word_fc(embeds)

        elif self.composer_method == 'bert':
            token_shape = tokens.shape[:-1]
            bert_input = tokens.reshape(token_shape.numel(), -1)
            bert_input_valid_mask = (bert_input.sum(dim=1) > 0)
            bert_input_valid = bert_input[bert_input_valid_mask]
            bert_attention_mask = (bert_input_valid != 0).float()
            embeds_valid = self.bert_embedding(bert_input_valid, bert_attention_mask)[0]
            embeds = torch.zeros(
                (bert_input.shape[0], bert_input.shape[1], embeds_valid.shape[-1]), requires_grad=True,
            ).to(tokens.device)

            N = bert_input.shape[0]
            indices =  torch.arange(N, device='cuda')
            indices = indices[bert_input_valid_mask]
            indices = indices.unsqueeze(1).unsqueeze(1).expand(-1, bert_input.shape[1], embeds_valid.shape[-1])
            embeds = embeds.scatter(dim=0, index=indices, src=embeds_valid)
            if self.bert_pooling == 'avg':
                bert_attention_mask = (bert_input != 0).float() # (N, L)
                embeds = (embeds * bert_attention_mask.unsqueeze(2)).sum(1) / (bert_attention_mask.sum(1, keepdim=True) + 1e-10)
                embeds = embeds.reshape(*token_shape, -1)
                embeds = self.word_fc(embeds)
            elif self.bert_pooling == 'cls':
                embeds = embeds[:, 0, :]
                embeds = embeds.reshape(*token_shape, -1)
                embeds = self.word_fc(embeds)
            else:
                # embeds: (bs, L, d)
                # bert_input: (bs, L)
                bert_input_lengths = (bert_input != 0).sum(1) # (bs)
                bert_input_lengths = bert_input_lengths.clamp(min=1)
                if token_type == 'entity':
                    embeds, _ = self.gpool_obj(embeds, bert_input_lengths)
                else:
                    embeds, _ = self.gpool_attr(embeds, bert_input_lengths)
                embeds = embeds.reshape(*token_shape, -1)
                embeds = self.word_fc(embeds)

        elif self.composer_method == 'bert-prefix':
            token_shape = tokens.shape[:-1]
            bert_input = tokens.reshape(token_shape.numel(), -1)
            bert_input_valid_mask = (bert_input.sum(dim=1) > 0)
            bert_input_valid = bert_input[bert_input_valid_mask]
            bert_attention_mask = (bert_input_valid != 0).float()
            embeds_valid = self.bert_prefix_model(bert_input_valid, bert_attention_mask)
            embeds = torch.zeros(
                (bert_input.shape[0], bert_input.shape[1], embeds_valid.shape[-1]), requires_grad=True,
            ).to(tokens.device)

            N = bert_input.shape[0]
            indices =  torch.arange(N, device='cuda')
            indices = indices[bert_input_valid_mask]
            indices = indices.unsqueeze(1).unsqueeze(1).expand(-1, bert_input.shape[1], embeds_valid.shape[-1])
            embeds = embeds.scatter(dim=0, index=indices, src=embeds_valid)
            if self.bert_pooling == 'avg':
                bert_attention_mask = (bert_input != 0).float() # (N, L)
                embeds = (embeds * bert_attention_mask.unsqueeze(2)).sum(1) / (bert_attention_mask.sum(1, keepdim=True) + 1e-9)
                embeds = embeds.reshape(*token_shape, -1)
                embeds = self.word_fc(embeds)
            elif self.bert_pooling == 'cls':
                embeds = embeds[:, 0, :]
                embeds = embeds.reshape(*token_shape, -1)
                embeds = self.word_fc(embeds)
            else:
                # embeds: (bs, L, d)
                # bert_input: (bs, L)
                bert_input_lengths = (bert_input != 0).sum(1) # (bs)
                bert_input_lengths = bert_input_lengths.clamp(min=1)
                if token_type == 'entity':
                    embeds, _ = self.gpool_obj(embeds, bert_input_lengths)
                else:
                    embeds, _ = self.gpool_attr(embeds, bert_input_lengths)
                embeds = embeds.reshape(*token_shape, -1)
                embeds = self.word_fc(embeds)

        return embeds


def build_entity_composer(cfg, vocab_size=-1):
    if cfg.MODEL.pretrained_word_embedding_path != '':
        pretrained_word_embedding = torch.load(cfg.MODEL.pretrained_word_embedding_path)
    else:
        pretrained_word_embedding = None

    return EntityComposer(
        vocab_size=vocab_size,
        word_embed_dim=cfg.MODEL.word_embed_dim,
        hidden_dim=cfg.MODEL.entity_composer_hidden_dim,
        embed_dim=cfg.MODEL.embed_dim,
        composer_n_layers=cfg.MODEL.entity_composer_n_layers,
        composer_method=cfg.MODEL.entity_composer_method,
        bert_pooling=cfg.MODEL.entity_bert_pooling,
        dropout=cfg.MODEL.entity_composer_dropout,
        pretrained_word_embedding=pretrained_word_embedding,
    )