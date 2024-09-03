import copy
import json
import numpy as np
import random
import torch


class EntitySampler:
    def __init__(self, cfg, vocab, tokenizer, tok2lemma):
        """Class to sample entity from scene graph and create input data for training CORA.

        Args:
        * cfg: config.
        * vocab: object of class Vocabulary.
        * tokenizer: used to tokenize object and attribute names into tokens.
        * tok2lemma: dictionary that maps from a token to its lemma (e.g., 'cars' -> 'car).
        """
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.tok2lemma = tok2lemma
        self.lemma2tok = {v:k for k,v in tok2lemma.items()}

        with open(f'{cfg.DATASET.directory}/vocab/rare_concepts.json', 'r') as f:
            self.rare_concepts = json.load(f)

        self.max_n_entities = cfg.TRAIN.max_n_entities
        self.max_n_entity_tokens = cfg.TRAIN.max_n_entity_tokens
        self.max_n_attributes = cfg.TRAIN.max_n_attributes
        self.max_n_attribute_tokens = cfg.TRAIN.max_n_attribute_tokens

        self.prob_drop_entity = cfg.TRAIN.prob_drop_entity
        self.prob_drop_attribute = cfg.TRAIN.prob_drop_attribute
        self.prob_singularize_obj = cfg.TRAIN.prob_singularize_obj
        self.prob_pluralize_obj = cfg.TRAIN.prob_pluralize_obj
        self.prob_mask_entity_token = cfg.TRAIN.prob_mask_entity_token
        self.prob_delete_entity_token = cfg.TRAIN.prob_delete_entity_token
        self.prob_replace_entity_token = cfg.TRAIN.prob_replace_entity_token

        if cfg.MODEL.word_embed_source.startswith('bert'):
            self.vocab_bert = list(tokenizer.vocab.keys())

    def sample_entities(self, sg, train=True):
        """Samples a subset of entities from scene graph.
        """
        sg = copy.deepcopy(sg)

        if train:
            # this is done only when training
            # randomly delete some entities from scene graph
            
            n_deleted = 0
            first_entity = sg['entities'][0]
            n_entities = len(sg['entities'])

            # delete from end to front
            for i in range(len(sg['entities'])-1, -1, -1):
                ent = sg['entities'][i]
                if np.random.rand() < self.prob_drop_entity and n_deleted < len(sg['entities'])-1:
                    n_deleted += 1
                    del sg['entities'][i]

            if n_deleted == n_entities:
                # if all entities are deleted -> add back the 1st entity to make the scene graph non-empty
                sg['entities'] = [first_entity]

            remaining_ids = set([ent['id'] for ent in sg['entities']])

            # keep only relations of the remaining entities
            relations = []
            for edge in sg['relations']:
                if edge[0] in remaining_ids and edge[1] in remaining_ids:
                    relations.append(edge)
            sg['relations'] = relations
        
        # truncate and keep only a maximum number of entities
        if len(sg['entities']) > self.max_n_entities:
            sg['entities'] = sg['entities'][:self.max_n_entities]

        for ent in sg['entities']:
            ent['attributes'] = self.sample_attributes(
                ent['attributes'], train=train
            )
            ent['name'] = ent['name'].lower()
            
            if train:
                if self.prob_singularize_obj > 0.0 and np.random.rand() < self.prob_singularize_obj:
                    # Singularize an object name. This helps train model to make correlation between singular and plural
                    # form of nouns.
                    if ent['name'] in self.tok2lemma:
                        ent['name'] = self.tok2lemma[ent['name']]
                    else:
                        toks = ent['name'].split()
                        if toks[-1] in self.tok2lemma:
                            toks[-1] = self.tok2lemma[toks[-1]]
                        ent['name'] = ' '.join(toks)
                else:
                    # Pluralize an object name.
                    if self.prob_pluralize_obj > 0.0 and np.random.rand() < self.prob_pluralize_obj:
                        if ent['name'] in self.lemma2tok:
                            ent['name'] = self.lemma2tok[ent['name']]
                        else:
                            toks = ent['name'].split()
                            if toks[-1] in self.lemma2tok:
                                toks[-1] = self.lemma2tok[toks[-1]]
                            ent['name'] = ' '.join(toks)

        return sg

    def sample_attributes(self, attributes, train=True):
        """Samples a subset of attributes.
        """
        if train:
            # only sample attributes when training

            selected_attrs = []
            selected_attrs_wo_advs = []

            for attr in attributes:
                # randomly remove some attributes
                if np.random.rand() < 1.0 - self.prob_drop_attribute:
                    selected_attrs.append(attr)

            # truncate and keep only a maximum number of attributes
            if len(selected_attrs) > self.max_n_attributes:
                np.random.shuffle(selected_attrs)
                selected_attrs = selected_attrs[:self.max_n_attributes]

            for i, attr in enumerate(selected_attrs):
                # each attribute also has adverbs associated with it
                advs = attr[2]

                # sample a subset of adverbs
                selected_advs = []
                for adv in advs:
                    if np.random.rand() < 1.0 - self.prob_drop_attribute:
                        selected_advs.append(adv)
                advs = ' '.join(selected_advs)

                if len(advs) > 0:
                    # append adverbs into attribute
                    selected_attrs[i] = f"{attr[1]} {advs}".lower()
                else:
                    selected_attrs[i] = attr[1].lower()
        else:
            selected_attrs = attributes

            # truncate and keep only a maximum number of attributes
            if len(selected_attrs) > self.max_n_attributes:
                selected_attrs = selected_attrs[:self.max_n_attributes]

            for i, attr in enumerate(selected_attrs):
                # each attribute also has adverbs associated with it
                advs = ' '.join(attr[2])

                if len(advs) > 0:
                    # append adverbs into attribute
                    selected_attrs[i] = f"{attr[1]} {advs}".lower()
                else:
                    selected_attrs[i] = attr[1].lower()

        return selected_attrs

    def construct_inputs(self, sg, train=True):
        """Examples of the entity's token sequence
        [
            [ [CLS], man, [SEP], [PAD], [PAD], ...],
            [ [CLS], construction, worker, [SEP], [PAD], [PAD], ...],
            ....
        ]
        """
        entities = sg['entities']
        entity_tokens = np.zeros((self.max_n_entities, self.max_n_entity_tokens), dtype=int)
        entity_lengths = np.zeros((self.max_n_entities,), dtype=int)
        entity_padding_mask = np.ones((self.max_n_entities, self.max_n_entity_tokens), dtype=bool)

        entity_attribute_tokens = np.zeros(
            (self.max_n_entities, self.max_n_attributes, self.max_n_attribute_tokens), dtype=int
        )
        entity_attribute_lengths = np.zeros((self.max_n_entities, self.max_n_attributes), dtype=int)
        entity_attribute_padding_mask = np.ones(
            (self.max_n_entities, self.max_n_attributes, self.max_n_attribute_tokens), dtype=bool
        )

        if self.cfg.MODEL.word_embed_source.startswith('bert'):
            cls_token_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
            sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
            pad_token_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        else:
            cls_token_id = self.vocab('[CLS]')
            sep_token_id = self.vocab('[SEP]')
            pad_token_id = self.vocab('[PAD]')
    
        for i, entity in enumerate(entities):
            # object tokens
            obj = entity['name']
            if self.cfg.MODEL.word_embed_source.startswith('bert'):
                tokens = self.tokenizer.tokenize(obj)
                tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                tokens = self.tokenizer(obj)
                tokens = [self.vocab(tok) for tok in tokens]

            tokens = self.mask_and_delete_tokens(tokens, train=train)
            tokens = [cls_token_id] + tokens + [sep_token_id]

            padding_mask = [False] * len(tokens)
            if len(tokens) > self.max_n_entity_tokens:
                tokens = tokens[:self.max_n_entity_tokens]
                tokens[-1] = sep_token_id
                padding_mask = padding_mask[:self.max_n_entity_tokens]
            entity_tokens[i,:len(tokens)] = tokens
            entity_padding_mask[i,:len(tokens)] = padding_mask
            entity_lengths[i] = len(tokens)

            # attribute tokens
            attributes = entity['attributes']
            for i_attr, attr in enumerate(attributes):
                attr = attr.lower()
                if self.cfg.MODEL.word_embed_source.startswith('bert'):
                    tokens = self.tokenizer.tokenize(attr)
                    tokens = self.tokenizer.convert_tokens_to_ids(tokens)
                else:
                    tokens = self.tokenizer(attr)
                    tokens = [self.vocab(tok) for tok in tokens]

                tokens = self.mask_and_delete_tokens(tokens, is_ent=False, train=train)
                tokens = [cls_token_id] + tokens + [sep_token_id]

                padding_mask = [False] * len(tokens)
                if len(tokens) > self.max_n_attribute_tokens:
                    tokens = tokens[:self.max_n_attribute_tokens]
                    tokens[-1] = sep_token_id
                    padding_mask = padding_mask[:self.max_n_attribute_tokens]
                entity_attribute_tokens[i, i_attr, :len(tokens)] = tokens
                entity_attribute_padding_mask[i, i_attr, :len(tokens)] = padding_mask
                entity_attribute_lengths[i, i_attr] = len(tokens)

        out = {
            'entity_tokens': torch.LongTensor(entity_tokens),
            'entity_lengths': torch.LongTensor(entity_lengths),
            'entity_padding_mask': torch.BoolTensor(entity_padding_mask),
            #######################
            'entity_attribute_tokens': torch.LongTensor(entity_attribute_tokens),
            'entity_attribute_padding_mask': torch.BoolTensor(entity_attribute_padding_mask),
            'entity_attribute_lengths': torch.LongTensor(entity_attribute_lengths),
        }
        
        return out

    def mask_and_delete_tokens(self, tokens, is_ent=True, train=True):
        """Data augmentaion on the text tokens.
        """
        if not train or self.prob_mask_entity_token <= 0.0:
            return tokens

        if self.cfg.MODEL.word_embed_source.startswith('bert'):
            mask_token_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        else:
            mask_token_id = self.vocab('[MASK]')

        orig_1st_token = random.choice(tokens)
        deleted_idx = []

        for i, token in enumerate(tokens):
            prob = np.random.rand()
            if prob < self.prob_mask_entity_token:
                # mask token
                tokens[i] = mask_token_id
            elif prob < self.prob_mask_entity_token + self.prob_delete_entity_token:
                # remove token
                deleted_idx.append(i)
            elif prob < self.prob_mask_entity_token + self.prob_delete_entity_token + self.prob_replace_entity_token:
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
