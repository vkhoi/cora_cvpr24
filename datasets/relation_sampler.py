import numpy as np
import random
import torch


class RelationSampler:
    def __init__(self, cfg, vocab, tokenizer, tok2lemma):
        """Class to sample relation from scene graph and create input data for training CORA.

        Args:
        * cfg: config.
        * vocab: object of class Vocabulary.
        * tokenizer: used to tokenize object and attribute names into tokens.
        * tok2lemma: dictionary that maps from a token to its lemma (e.g., 'cars' -> 'car').
        """
        self.cfg = cfg
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.tok2lemma = tok2lemma

        self.max_n_entities = cfg.TRAIN.max_n_entities
        self.max_n_relations = cfg.TRAIN.max_n_relations
        self.max_n_relation_tokens= cfg.TRAIN.max_n_relation_tokens

        self.prob_drop_relation = cfg.TRAIN.prob_drop_relation 
        self.prob_mask_relation_token = cfg.TRAIN.prob_mask_relation_token
        self.prob_delete_relation_token = cfg.TRAIN.prob_delete_relation_token
        self.prob_replace_relation_token = cfg.TRAIN.prob_replace_relation_token

        if cfg.MODEL.word_embed_source.startswith('bert'):
            self.vocab_bert = list(tokenizer.vocab.keys())

    def sample_relations(self, scene_graph, train=True):
        """Samples a subset of relations from a given scene graph.
        """
        if train:
            selected_relations = []
            for rel in scene_graph['relations']:
                if np.random.rand() < 1.0 - self.prob_drop_relation:
                    selected_relations.append(rel)
            np.random.shuffle(selected_relations)
        else:
            selected_relations = scene_graph['relations']
        scene_graph['relations'] = selected_relations
        return scene_graph

    def construct_inputs(self, sg, train=True):
        # Record position of each entity in the entity list
        pos_of_eid = {}
        for i in range(len(sg['entities'])):
            eid = sg['entities'][i]['id']
            pos_of_eid[eid] = i

        exist_edge = set()
        for rel in sg['relations']:
            subj, obj = rel[0], rel[1]
            exist_edge.add((subj, obj))

        # For a graph with n entities, we consider it to have n+1 nodes, where node 0 is the global node that connects
        # to all entities, and nodes 1->n are the entities in the graph.

        n_nodes = len(sg['entities'])
        relation_adj_matrix = np.zeros((self.max_n_entities+1, self.max_n_entities+1), dtype=float)
        
        # global node is connected to all entities
        relation_adj_matrix[0,1:n_nodes+1] = 1

        # a node is connected to itself
        for i in range(n_nodes+1):
            relation_adj_matrix[i,i] = 1

        # to store the active relation tokens for an entity
        relation_sbj_tokens = np.zeros((self.max_n_entities, self.max_n_relations, self.max_n_relation_tokens), dtype=int)
        relation_sbj_lengths = np.zeros((self.max_n_entities, self.max_n_relations), dtype=int)
        relation_sbj_padding_mask = np.ones((self.max_n_entities, self.max_n_relations, self.max_n_relation_tokens), dtype=bool)
        relation_sbj_obj_ids = np.ones((self.max_n_entities, self.max_n_relations), dtype=int) * (self.max_n_entities-1)

        # to store the passive relation tokens for an entity
        relation_obj_tokens = np.zeros((self.max_n_entities, self.max_n_relations, self.max_n_relation_tokens), dtype=int)
        relation_obj_lengths = np.zeros((self.max_n_entities, self.max_n_relations), dtype=int)
        relation_obj_padding_mask = np.ones((self.max_n_entities, self.max_n_relations, self.max_n_relation_tokens), dtype=bool)
        relation_obj_sbj_ids = np.ones((self.max_n_entities, self.max_n_relations), dtype=int) * (self.max_n_entities-1)

        sbj_rel_idx = {x: 0 for x in range(self.max_n_entities)}
        obj_rel_idx = {x: 0 for x in range(self.max_n_entities)}

        if self.cfg.MODEL.word_embed_source.startswith('bert'):
            cls_token_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
            sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
            pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        else:
            cls_token_id = self.vocab('[CLS]')
            sep_token_id = self.vocab('[SEP]')
            pad_token_id = self.vocab('[PAD]')

        for ent in sg['entities']:
            # for every entity
            subj_id = ent['id']
            u = pos_of_eid[subj_id]

            # get all active relations of the current entity
            edges = [edge for edge in sg['relations'] if edge[0] == subj_id and edge[1] in pos_of_eid]

            for edge in edges:
                # find the object of this relation
                obj_id = edge[1]

                # get the name of this relation
                acts = edge[2]
                rel = ' '.join(acts)

                pos = edge[3]
                if len(pos) > 0:
                    rel += ' ' + pos
                if len(rel) == 0:
                    rel = random.choice(['and', 'with', ''])

                if self.cfg.MODEL.word_embed_source.startswith('bert'):
                    rel_tokens = self.tokenizer.tokenize(rel)
                    rel_tokens = self.tokenizer.convert_tokens_to_ids(rel_tokens)
                else:
                    rel_tokens = self.tokenizer(rel)
                    rel_tokens = [self.vocab(tok) for tok in rel_tokens]

                if train and self.prob_mask_relation_token > 0.0 and len(rel_tokens) > 0:
                    self.mask_and_delete_tokens(rel_tokens, train=train)

                rel_tokens = [cls_token_id] + rel_tokens + [sep_token_id]

                v = pos_of_eid[edge[1]]
                relation_adj_matrix[u+1, v+1] = 1 # forward edge
                relation_adj_matrix[v+1, u+1] = 1 # reverse edge

                # Update for gcn
                self.update_edge_sbj_obj_rel(
                    u, v, rel_tokens,
                    sbj_rel_idx, relation_sbj_tokens, relation_sbj_padding_mask, relation_sbj_obj_ids, relation_sbj_lengths,
                    obj_rel_idx, relation_obj_tokens, relation_obj_padding_mask, relation_obj_sbj_ids, relation_obj_lengths,
                    sep_token_id)

        relation_global_adj_matrix = np.zeros((self.max_n_entities+1, self.max_n_entities+1), dtype=float)
        relation_global_adj_matrix[:n_nodes+1,:n_nodes+1] = 1

        out = {
            'relation_sbj_tokens': torch.LongTensor(relation_sbj_tokens),
            'relation_sbj_padding_mask': torch.BoolTensor(relation_sbj_padding_mask),
            'relation_sbj_obj_ids': torch.LongTensor(relation_sbj_obj_ids),
            'relation_sbj_lengths': torch.LongTensor(relation_sbj_lengths),
            #######################
            'relation_obj_tokens': torch.LongTensor(relation_obj_tokens),
            'relation_obj_padding_mask': torch.BoolTensor(relation_obj_padding_mask),
            'relation_obj_sbj_ids': torch.LongTensor(relation_obj_sbj_ids),
            'relation_obj_lengths': torch.LongTensor(relation_obj_lengths),
            #######################,
            'relation_adj_matrix': torch.FloatTensor(relation_adj_matrix),
            'relation_global_adj_matrix': torch.FloatTensor(relation_global_adj_matrix),
        }
        
        return out

    def mask_and_delete_tokens(self, tokens, train=True):
        """Data augmentaion on the text tokens.
        """
        if not train or self.prob_mask_relation_token <= 0.0:
            return tokens

        if self.cfg.MODEL.word_embed_source.startswith('bert'):
            mask_token_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        else:
            mask_token_id = self.vocab('[MASK]')

        orig_1st_token = random.choice(tokens)
        deleted_idx = []

        for i, token in enumerate(tokens):
            prob = np.random.rand()
            if prob < self.prob_mask_relation_token:
                # mask token
                tokens[i] = mask_token_id
            elif prob < self.prob_mask_relation_token + self.prob_delete_relation_token:
                # remove token
                deleted_idx.append(i)
            elif prob < self.prob_mask_relation_token + self.prob_delete_relation_token + self.prob_replace_relation_token:
                # replace token
                if self.cfg.MODEL.word_embed_source.startswith('bert'):
                    tokens[i] = np.random.randint(len(self.vocab_bert))
                else:
                    tokens[i] = int(np.random.choice(self.vocab.all_tokens))
        
        # keep only tokens that are not removed
        tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]

        # if all tokens are masked then restore the first token
        if len(tokens) == 0 or all([x == mask_token_id for x in tokens]):
            tokens = [orig_1st_token]

        return tokens

    def update_edge_sbj_obj_rel(
        self, subj_idx, obj_idx, rel_tokens,
        sbj_rel_idx, sbj_rel_tokens, sbj_rel_padding_mask, sbj_obj_ids, sbj_rel_lengths,
        obj_rel_idx, obj_rel_tokens, obj_rel_padding_mask, obj_sbj_ids, obj_rel_lengths,
        sep_token_id
    ):
        # truncate to max number of tokens for relations        
        rel_len = min(len(rel_tokens), self.max_n_relation_tokens)
        rel_tokens[rel_len-1] = sep_token_id

        # find the index for the active edge of this relation
        edge_idx = sbj_rel_idx[subj_idx]

        # add this relation if we still not exceed the max num of relations
        if edge_idx < self.max_n_relations:
            # track the number relations have been added for this subj_idx
            sbj_rel_idx[subj_idx] += 1

            # fill in the tokens of this relation
            sbj_rel_tokens[subj_idx, edge_idx, :rel_len] = rel_tokens[:rel_len]

            # set the padding mask (which are padded tokens)
            sbj_rel_padding_mask[subj_idx, edge_idx, :rel_len] = False

            # what is the id of the object on the receiving end of the relation
            sbj_obj_ids[subj_idx, edge_idx] = obj_idx

            # number of tokens of this relation
            sbj_rel_lengths[subj_idx, edge_idx] = rel_len
        
        # find the index for the passive edge of this relation
        edge_idx = obj_rel_idx[obj_idx]

        # do the same things above but for the passive egde
        if edge_idx < self.max_n_relations:
            obj_rel_idx[obj_idx] += 1
            obj_rel_tokens[obj_idx, edge_idx, :rel_len] = rel_tokens[:rel_len]
            obj_rel_padding_mask[obj_idx, edge_idx, :rel_len] = False
            obj_sbj_ids[obj_idx, edge_idx] = subj_idx
            obj_rel_lengths[obj_idx, edge_idx] = rel_len
