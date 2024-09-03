import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pack_padded_sequence

# from models.modules.bert_prefix_model import BertPrefixModel
from models.modules.gpo import GPO
from models.modules.graph_attention import GraphAttention


class RelationComposer(nn.Module):
    def __init__(
        self,
        vocab_size=-1,
        word_embed_dim=300,
        hidden_dim=512,
        embed_dim=1024,
        composer_n_layers=3,
        composer_method='gru',
        bert_pooling='cls',
        dropout=0.0,
        graph_pos_embed_dim=0,
        use_graph_global=False,
    ):
        """Initializes relation composer class (compose objects with objects through their relationships).

        Args:
        * vocab_size: if not using BERT, this is the size of the vocabulary parsed from the training set. Otherwise, -1.
        * word_embed_dim: if not using BERT, this is the word embedding dimension.
        * hidden_dim: intermediate hidden dimension of the graph attention network.
        * embed_dim: dimension of the embedding space.
        * composer_n_layers: number of layers in the graph attention network.
        * composer_method: what kind of text model to embed the object and attribute nodes.
        * bert_pooling: if using BERT, what kind of pooling method to apply on the output text embeddings.
        * dropout: dropout rate to use in graph attention network.
        * graph_pos_embed_dim: dimension of position embedding for graph nodes. Why use position embedding here?
        Position embedding seems to show which nodes are more important because usually, the first few nodes (parsed
        early from the text) are more salient.
        * use_graph_global: whether to use global graph (i.e., all nodes are fully connected with each other).
        """
        super().__init__()
        if vocab_size == -1:
            vocab_size = 30522
        
        if composer_method not in ['bert', 'bert-prefix']:
            self.word_embeddings = nn.Embedding(vocab_size, word_embed_dim)

        self.composer_method = composer_method
        if composer_method == 'gru':
            self.rnn_relation = nn.GRU(word_embed_dim, word_embed_dim, num_layers=1, batch_first=True, bidirectional=False)
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

        # output embeddings from entity graph can have different dimensions, so we need to have linear layer
        # here if that this the case.
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        if embed_dim != hidden_dim:
            self.entity_fc = nn.Linear(embed_dim, hidden_dim)

        # position embeddings for the graph nodes
        self.graph_pos_embed_dim = graph_pos_embed_dim
        if graph_pos_embed_dim > 0:
            self.pos_linear = nn.Linear(graph_pos_embed_dim, hidden_dim)
            self.dropout_pos = nn.Dropout(dropout)
            self.pos_lnorm = nn.LayerNorm(hidden_dim)

        self.composer_n_layers = composer_n_layers
        self.drop_input = nn.Dropout(dropout)
        self.drop_input_global = nn.Dropout(dropout)
        self.drop_output = nn.Dropout(dropout)
        self.drop_output_global = nn.Dropout(dropout)
        self.graph_attention = GraphAttention(
            n_layers=composer_n_layers, d=hidden_dim, dropout=0.05,
        )
        if hidden_dim != embed_dim:
            self.global_out = nn.Linear(hidden_dim, embed_dim)

        # FC layer to embed active and passage edges
        self.active_edge = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        self.passive_edge = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )
        self.lnorm = nn.LayerNorm(hidden_dim)
        self.gpool = GPO(32, 32)

        self.use_graph_global = use_graph_global
        if use_graph_global:
            self.graph_global = GraphAttention(
                n_layers=composer_n_layers, d=hidden_dim, dropout=0.05,
            )
            self.gpool_full = GPO(32, 32)

    def set_word_embeddings_weight(self, weight):
        self.word_embeddings.weight = weight
    
    def set_bert_model(self, bert_embedding):
        if hasattr(self, 'bert_embedding'):
            self.bert_embedding = bert_embedding
        elif hasattr(self, 'bert_prefix_model'):
            self.bert_prefix_model = bert_embedding

    def embed_text_phrase(self, tokens, lengths=None):
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
            _, embeds = self.rnn_relation(packed)
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
                embeds, _ = self.gpool_rel(embeds, bert_input_lengths)
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

    def forward(self, batch, entity_embeds):
        # entity_embeds: (bs, max_n_ents, d)

        relation_sbj_tokens = batch['relation_sbj_tokens']
        relation_sbj_padding_mask = batch['relation_sbj_padding_mask'] # (bs, max_n_ents, max_n_rels, max_n_rel_tokens)
        relation_sbj_obj_ids = batch['relation_sbj_obj_ids'] # (bs, max_n_ents, max_n_rels)
        relation_sbj_lengths = batch['relation_sbj_lengths'] # (bs, max_n_ents, max_n_rels)
        relation_obj_tokens = batch['relation_obj_tokens']
        relation_obj_padding_mask = batch['relation_obj_padding_mask']
        # relation_obj_sbj_ids = batch[f'{prefix}_obj_sbj_ids']
        relation_obj_lengths = batch['relation_obj_lengths']
        relation_adj_matrix = batch['relation_adj_matrix']

        if self.embed_dim != self.hidden_dim:
            # check if output embeddings from entity graph have different dimensions
            entity_embeds = self.entity_fc(entity_embeds)

        if self.graph_pos_embed_dim > 0:
            # check if need to apply position embedding for nodes in the graph
            pos_embeds = batch['graph_pos_embeds']
            pos_embeds = self.pos_linear(pos_embeds) # (bs, max_n_ents, d)
            pos_embeds = self.dropout_pos(pos_embeds)
            entity_embeds = entity_embeds + pos_embeds
            entity_embeds = self.pos_lnorm(entity_embeds)

        bs = entity_embeds.shape[0]
        max_n_ents = entity_embeds.shape[1]
        entity_dim = entity_embeds.shape[-1]
        max_n_rels = relation_sbj_obj_ids.shape[-1]

        # embed text tokens of the active relations
        sbj_rel_embeds = self.embed_text_phrase(
            relation_sbj_tokens, relation_sbj_lengths
        ) # (bs, max_n_ents, max_n_rels, d)
        # gather embeddings of the object nodes
        obj_embeds = torch.gather(
            entity_embeds.unsqueeze(2).expand(-1, -1, max_n_rels, -1),
            dim=1,
            index=relation_sbj_obj_ids.unsqueeze(-1).expand(-1, -1, -1, entity_dim))
        # concat active relations with corresponding object nodes
        sbj_rel_embeds = torch.cat((sbj_rel_embeds, obj_embeds), dim=-1) # (bs, max_n_ents, max_n_rels, d*3)
        # run it through active egde FC
        sbj_rel_embeds = self.active_edge(sbj_rel_embeds) # (bs, max_n_ents, max_n_rels, d)

        # embed text tokens of the passive relations
        obj_rel_embeds = self.embed_text_phrase(relation_obj_tokens, relation_obj_lengths) # (bs, max_n_ents, max_n_rels, d)
        # gather embeddings of the object nodes
        obj_embeds = entity_embeds.unsqueeze(2).expand_as(obj_rel_embeds)
        # concat passive relations with corresponding object nodes
        obj_rel_embeds = torch.cat((obj_rel_embeds, obj_embeds), dim=-1) # (bs, max_n_ents, max_n_rels, d*3)
        # run it through passive egde FC
        obj_rel_embeds = self.passive_edge(obj_rel_embeds) # (bs, max_n_ents, max_n_rels, d)

        # subject role
        sbj_rel_embeds = sbj_rel_embeds.view(bs * max_n_ents, max_n_rels, -1)
        sbj_atten_mask = (~relation_sbj_padding_mask)[:,:,:,0].view(bs * max_n_ents, -1)

        # object role
        obj_rel_embeds = obj_rel_embeds.view(bs * max_n_ents, max_n_rels, -1)
        obj_atten_mask = (~relation_obj_padding_mask)[:,:,:,0].view(bs * max_n_ents, -1)

        # merge by taking average of all relations
        obj_embeddings = torch.cat([sbj_rel_embeds, obj_rel_embeds], dim=1)
        obj_mask = torch.cat([sbj_atten_mask, obj_atten_mask], dim=1)
        obj_embeddings = (obj_embeddings * obj_mask.unsqueeze(2)).sum(1) / (obj_mask.sum(1, keepdim=True) + 1e-8)

        # residual connection
        obj_embeddings = obj_embeddings.view(bs, max_n_ents, -1)
        entity_embeds = entity_embeds + obj_embeddings
        entity_embeds = self.lnorm(entity_embeds)

        # graph attention network
        entity_embeds = self.drop_input(entity_embeds)
        relation_adj_matrix = relation_adj_matrix[:,1:,1:]
        for t in range(self.composer_n_layers):
            entity_embeds = self.graph_attention(entity_embeds, relation_adj_matrix, t=t)
            if t < self.composer_n_layers - 1:
                entity_embeds = self.drop_output(entity_embeds)
        entity_length = (relation_adj_matrix.sum(-1) > 0).sum(-1) # (bs)
        graph_embed, _ = self.gpool(entity_embeds, entity_length)

        if self.use_graph_global:
            # if use global graph
            entity_embeds_full = entity_embeds
            relation_global_adj_matrix = batch['relation_global_adj_matrix']
            relation_global_adj_matrix = relation_global_adj_matrix[:,1:,1:]
        
            for t in range(self.composer_n_layers):
                entity_embeds_full = self.graph_global(entity_embeds_full, relation_global_adj_matrix, t=t)
                if t < self.composer_n_layers - 1:
                    entity_embeds_full = self.drop_output(entity_embeds_full)
            graph_embed_full, _ = self.gpool_full(entity_embeds_full, entity_length)
            graph_embed = graph_embed + graph_embed_full

        if self.embed_dim != self.hidden_dim:
            graph_embed = self.global_out(graph_embed)

        return graph_embed


def build_relation_composer(cfg, vocab_size=-1):
    return RelationComposer(
        vocab_size=vocab_size,
        word_embed_dim=cfg.MODEL.word_embed_dim,
        hidden_dim=cfg.MODEL.rel_composer_hidden_dim,
        embed_dim=cfg.MODEL.embed_dim,
        composer_n_layers=cfg.MODEL.rel_composer_n_layers,
        composer_method=cfg.MODEL.rel_composer_method,
        bert_pooling=cfg.MODEL.rel_bert_pooling,
        dropout=cfg.MODEL.rel_composer_dropout,
        graph_pos_embed_dim=cfg.MODEL.graph_pos_embed_dim,
        use_graph_global=cfg.MODEL.use_graph_global,
    )