import cv2
import hashlib
import json
import math
import numpy as np
import os
import torch
import torch.utils.data as data

from PIL import Image

from datasets.entity_sampler import EntitySampler
from datasets.relation_sampler import RelationSampler
from datasets.vocab import get_vocab


class Example:
    def __init__(self, image_id, caption, image_path, scene_graph, img_index):
        self.image_id = image_id
        self.caption = caption
        self.image_path = image_path
        self.scene_graph = scene_graph
        self.img_index = img_index


class BaseDataset:
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.split = split

        self.img_dataset_path = f'{cfg.DATASET.directory}/images'
        with open(f'{cfg.DATASET.directory}/id_mapping.json', 'r') as f:
            self.id_to_imgpath = json.load(f)

        # read captions
        with open(f'{cfg.DATASET.directory}/{split}_caps.txt', 'r') as f:
            self.captions = [l.strip() for l in f.readlines()]
        # read image ids
        with open(f'{cfg.DATASET.directory}/{split}_ids.txt', 'r') as f:
            self.image_ids = [l.strip() for l in f.readlines()]
        # read scene graphs
        with open(f'{cfg.DATASET.directory}/scene_graph/{split}.json', 'r') as f:
            self.scene_graph = json.load(f)
        with open(f'{cfg.DATASET.directory}/vocab/tok2lemma.json', 'r') as f:
            self.tok2lemma = json.load(f)

        vocab, tokenizer = get_vocab(f'{cfg.DATASET.directory}/vocab/vocab.json', ty=cfg.MODEL.word_embed_source)
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.n_captions = len(self.captions)
        self.n_images = len(self.image_ids)
        if self.n_captions != self.n_images:
            self.im_div = 5
        else:
            self.im_div = 1

        self.examples = []
        for i in range(len(self.captions)):
            img_index = i // self.im_div
            image_id = self.image_ids[img_index]
            sg = self.scene_graph[str(i)]
            sg['index'] = i
            ex = Example(
                image_id=image_id,
                caption=sg['caption'],
                image_path=f"{self.img_dataset_path}/{self.id_to_imgpath[str(image_id)]}",
                scene_graph=sg,
                img_index=img_index,
            )
            self.examples.append(ex)

        if split == 'dev':
            self.examples = self.examples[:5000]

class ImageCaptionDataset(data.Dataset):
    def __init__(self, dataset: BaseDataset, train=False):
        self.dataset = dataset
        self.cfg = dataset.cfg
        self.tokenizer = dataset.tokenizer
        self.vocab = dataset.vocab
        self.train = train

        self.entity_sampler = EntitySampler(
            self.cfg, self.vocab, self.tokenizer, self.dataset.tok2lemma,
        )
        self.relation_sampler = RelationSampler(
            self.cfg, self.vocab, self.tokenizer, self.dataset.tok2lemma,
        )
        
        if self.cfg.MODEL.img_encoder_type == 'grid':
            self.image_size = 512
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.dataset.examples)

    def __getitem__(self, index):
        example = self.dataset.examples[index]
        sg = example.scene_graph
        image_path = example.image_path
        img_index = example.img_index
        out = {
            'index': index,             # index of this example
            'img_index': img_index,     # index of the image of this example
        }

        if self.cfg.MODEL.img_encoder_type == 'region':
            # read region feature file
            region_file_idx = index // self.dataset.im_div
            region_file_part = region_file_idx // 5000
            regions = np.load(
                f'{self.cfg.DATASET.directory}/region_feats/'
                f'{self.dataset.split}/part{region_file_part}/{region_file_idx}.npy')

            if self.train and self.cfg.TRAIN.region_dropout:
                # perform region dropout
                regions_tensor, region_lengths = self._region_dropout(regions)
                regions_tensor = torch.FloatTensor(regions_tensor)
                
                # mask to show which are the padded regions
                regions_padding_mask = ([0] * region_lengths) + ((regions_tensor.shape[0] - region_lengths) * [1])
                regions_padding_mask = torch.BoolTensor(regions_padding_mask)
                
                out.update({
                    'img': regions_tensor,
                    'img_length': region_lengths,
                    'img_padding_mask': regions_padding_mask
                })

                if self.cfg.MODEL.img_pooling_type == 'gpo-selfattn':
                    # position embeddings for the regions
                    d_pe = self.cfg.MODEL.img_pos_embed_dim
                    pos_embeds = positional_encoding_1d(d_pe, region_lengths, device='cpu')
                    if region_lengths < 36:
                        pos_embeds = torch.cat((pos_embeds, torch.zeros(36-region_lengths, d_pe)), dim=0)
                    out.update({
                        'img_pos_embeds': pos_embeds
                    })
            else:
                # when testing or not doing region dropout, we keep all regions
                regions_tensor = torch.FloatTensor(regions)
                out.update({
                    'img': regions_tensor,
                    'img_length': regions_tensor.shape[0],
                })
                if self.cfg.MODEL.img_pooling_type == 'gpo-selfattn':
                    # position embeddings for the regions
                    d_pe = self.cfg.MODEL.img_pos_embed_dim
                    pos_embeds = positional_encoding_1d(d_pe, 36, device='cpu')
                    out.update({
                        'img_pos_embeds': pos_embeds
                    })
        else:
            im = np.array(Image.open(image_path).convert('RGB'))
            processed_image = self._process_image(im)
            img = torch.Tensor(processed_image)
            img = img.permute(2, 0, 1)
            out.update({
                'img': img
            })

        # sample subset of entities from scene graph
        sg_sampled = self.entity_sampler.sample_entities(sg, train=self.train)
        out_entity = self.entity_sampler.construct_inputs(sg_sampled, train=self.train)
        out.update(out_entity)

        # sample subset of relations (edges) from scene graph for the above sampled subset of entities
        sg_sampled = self.relation_sampler.sample_relations(sg_sampled, train=self.train)
        out_relation = self.relation_sampler.construct_inputs(sg_sampled, train=self.train)
        out.update(out_relation)

        # whether to add position embeddings for the graph nodes
        d_pe = self.cfg.MODEL.graph_pos_embed_dim
        if d_pe > 0:
            n_entities = len(sg_sampled['entities'])
            graph_pos_embeds = positional_encoding_1d(d_pe, n_entities, device='cpu')
            if n_entities < self.cfg.TRAIN.max_n_entities:
                graph_pos_embeds = torch.cat(
                    (graph_pos_embeds, torch.zeros(self.cfg.TRAIN.max_n_entities-n_entities, d_pe)), 
                    dim=0
                )
            out.update({
                'graph_pos_embeds': graph_pos_embeds
            })

        if self.train:
            # these will be used in collate_fn
            out['sg_sampled'] = sg_sampled
            sg_original = self.entity_sampler.sample_entities(sg, train=False)
            out['sg'] = sg_original

        return out

    def _region_dropout(self, regions):
        """
        - regions: (36, d)
        """
        n_regions, feat_size = regions.shape

        p1 = np.random.rand()
        if p1 > self.cfg.TRAIN.region_dropout_prob1:
            # not performing any dropout
            return regions, n_regions

        probs = np.random.rand(n_regions)
        selected_regions = regions[np.where(probs > self.cfg.TRAIN.region_dropout_prob2)]
        n_selected_regions = len(selected_regions)
        padded_regions = torch.zeros((n_regions - n_selected_regions, feat_size))
        selected_regions = torch.FloatTensor(selected_regions)
        selected_regions = torch.cat((selected_regions, padded_regions), dim=0)

        return selected_regions, n_selected_regions

    def _process_image(self, im_in):
        """Converts an image into a network input, with pre-processing including re-scaling, padding, etc, and data
        augmentation.
        """
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        im = im_in.astype(np.float32, copy=True)
        image_size = self.image_size

        # 2. Random crop when in training mode, elsewise just skip
        if self.train:
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im = self._crop(im, crop_size_h, crop_size_w, random=True)
        else:
            processed_im = im

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(image_size) / im_shape[1]
        im_scale_y = float(image_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        if self.train:
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)

        # Normalization
        processed_im = self._imagenet_norm(processed_im)

        return processed_im

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in

    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im


def positional_encoding_1d(d_model, length, device):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def collate_fn(data):
    """This function finds all (image, entity) positive pairs to mask them out in the contrastive loss.

    An image I_i has positive match with its own entities {e_ij}. Apart from I_i, there can be other images in the
    training batch that are also positive with the entities {e_ij}. To find all (image, entity) positive pairs, we
    can traverse over their scene graphs and do text matching to check if an entity exists in an image's scene graph.
    """

    # stack every thing into Torch tensor
    out = {}
    for ex in data:
        for k, v in ex.items():
            if k not in out:
                out[k] = []
            out[k].append(v)
    for k in out:
        if isinstance(out[k][0], torch.Tensor):
            out[k] = torch.stack(out[k], dim=0)
    if 'img_length' in out:
        out['img_length'] = torch.LongTensor(out['img_length'])
    if 'img_index' in out:
        out['img_index'] = torch.LongTensor(out['img_index'])

    # find the total number of entities in the batch
    n_images = out['img'].shape[0]
    n_entities = 0
    for sg in out['sg_sampled']:
        n_entities += len(sg['entities'])

    MAX_N_ENTITIES = n_images * 6 # we make the pos_mask slightly larger than it actually is needed to be
    img_entity_pos_mask = torch.zeros((n_images, MAX_N_ENTITIES)).long()

    for i in range(n_images):
        # consider image i with its scene graph
        sg1 = out['sg'][i]
        pos = 0

        for j in range(n_images):
            # consider image j with its scene graph
            sg2 = out['sg_sampled'][j]

            for j_ent in range(len(sg2['entities'])):
                # iterate over each entity in scene graph j
                j_ent = sg2['entities'][j_ent]

                for i_ent in range(len(sg1['entities'])):
                    # iterate over each entity in scene graph i
                    i_ent = sg1['entities'][i_ent]
                    
                    # attributes of this entity
                    i_attrs = set(i_ent['attributes'])

                    # check if we have a match between the entity names and their attributes
                    if j_ent['name'] == i_ent['name']:
                        j_attrs = set(j_ent['attributes'])
                        if len(i_attrs & j_attrs) == len(j_attrs):
                            # it's a match => set this (image, entity) pair to positive
                            img_entity_pos_mask[i,pos] = 1
                            break
                
                # move on to the next entity
                pos += 1

        assert pos == n_entities

    img_entity_pos_mask[:,n_entities:] = -1
    out['img_entity_pos_mask'] = img_entity_pos_mask

    return out
