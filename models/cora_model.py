import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.losses.supcon import supcon_joint_caption_entity_loss
from models.losses.triplet import triplet_specificity_loss, image_caption_triplet_loss

from models.image_encoder import build_image_encoder
from models.entity_composer import build_entity_composer
from models.relation_composer import build_relation_composer

from models.modules.gather_layer import GatherLayer


class CoraModel(nn.Module):
    """Compositional Object from Relation and Attribute
    """
    def __init__(
        self,
        image_encoder,
        entity_composer,
        relation_composer,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.entity_composer = entity_composer
        self.relation_composer = relation_composer
    
    def setup_training_loss(self, cfg):
        self.batch_per_gpu = cfg.TRAIN.batch_size
        self.n_gpus = cfg.DISTRIBUTED.world_size

        # temperature for contrastive loss
        self.contrastive_temperature = cfg.TRAIN.contrastive_temperature

        # loss weight term and margin for the specificity triplet loss
        self.specificity_loss_weight = cfg.TRAIN.specificity_loss_weight
        self.specificity_margin = cfg.TRAIN.specificity_margin

        # loss weight term and margin for the vse++ loss
        self.image_caption_hardest_loss_weight = cfg.TRAIN.image_caption_hardest_loss_weight
        self.hardest_margin = cfg.TRAIN.hardest_margin

        if cfg.DISTRIBUTED.world_size > 1:
            self.is_distributed = True
        else:
            self.is_distributed = False

    def forward(self, batch, epoch=0, debug=False):
        img_embeds = self.image_encoder(batch)
        entity_embeds = self.entity_composer(batch)
        caption_embeds = self.relation_composer(batch, entity_embeds)

        img_embeds = F.normalize(img_embeds, dim=-1)
        entity_embeds = F.normalize(entity_embeds, dim=-1)
        caption_embeds = F.normalize(caption_embeds, dim=-1)

        out = {
            'img_embeds': img_embeds,
            'ent_embeds': entity_embeds,
            'cap_embeds': caption_embeds,
        }

        if self.training:
            out.update(self.forward_train(out, batch, epoch))

        return out

    def forward_train(self, out, batch, epoch):
        """Computes training losses using the output from model's forward method.
        """
        img_embeds = out['img_embeds'] # (bs, d)
        cap_embeds = out['cap_embeds'] # (bs, d)
        ent_embeds = out['ent_embeds'] # (bs, max_n_entities, d)

        entity_valid_mask = batch['entity_valid_mask'] # batch['entity_valid_mask'] is set by entity_composer
        bs, max_n_entities = entity_valid_mask.shape
        device = img_embeds.device

        out = {}
        if 'img_length' in batch:
            # When using region features, this is the number of regions after dropout. When using grid features,
            # this is the number of feature vectors in the feature map after dropout.

            # track the average number of regions after dropout
            if self.is_distributed:
                img_length = torch.cat(GatherLayer.apply(batch['img_length']), dim=0)
                out['n_regions_after_dropout'] = img_length.float().mean()
            else:
                out['n_regions_after_dropout'] = batch['img_length'].float().mean()

        # Gather embeddings across all GPUs (don't have to do this if we use DataParallel).
        # Source: https://github.com/openai/CLIP/issues/132
        img_indices = None
        if self.is_distributed:
            all_img_embeds = torch.cat(GatherLayer.apply(img_embeds), dim=0) # (bs, d)
            all_cap_embeds = torch.cat(GatherLayer.apply(cap_embeds), dim=0) # (bs, d)
            all_ent_embeds = torch.cat(GatherLayer.apply(ent_embeds), dim=0) # (bs, max_n_entities, d)
            all_ent_valid_mask = torch.cat(GatherLayer.apply(entity_valid_mask), dim=0) # (bs, max_n_entities)
            bs = all_img_embeds.shape[0]
            all_img_entity_pos_mask = torch.cat(GatherLayer.apply(batch['img_entity_pos_mask']), dim=0)
            if 'img_index' in batch:
                img_indices = torch.cat(GatherLayer.apply(batch['img_index']), dim=0) # To reduce noise for hardest triplet loss
        else:
            all_img_embeds = img_embeds
            all_cap_embeds = cap_embeds
            all_ent_embeds = ent_embeds
            all_ent_valid_mask = entity_valid_mask
            all_img_entity_pos_mask = batch['img_entity_pos_mask']
            if 'img_index' in batch:
                img_indices = batch['img_index']
        out['all_img_embeds'] = all_img_embeds
        out['all_cap_embeds'] = all_cap_embeds

        all_ent_embeds = all_ent_embeds[all_ent_valid_mask] # (n, d)
        out['all_ent_embeds'] = all_ent_embeds

        # create list of image indices for all entities (i.e., which image does an entity belong to)
        all_ent_img_ids = torch.arange(bs, device=device).unsqueeze(1).expand(-1, max_n_entities)
        all_ent_img_ids = all_ent_img_ids[all_ent_valid_mask] # (n_ent)
        out['all_ent_img_ids'] = all_ent_img_ids
        # all_ent_img_ids[i] = index of the image the entity i belongs to

        all_img_entity_pos_mask = self.prepare_image_entity_mask(all_img_entity_pos_mask)
        out['all_img_entity_pos_mask'] = all_img_entity_pos_mask

        contrastive_losses = supcon_joint_caption_entity_loss(
            all_img_embeds, all_cap_embeds, all_ent_embeds, all_ent_img_ids,
            temperature=self.contrastive_temperature,
            img_ent_pos_mask=all_img_entity_pos_mask,
            img_indices=img_indices,
        )

        total_loss = contrastive_losses['total_loss']
        out.update({
            'image_to_caption_loss': contrastive_losses['image_to_caption_loss'],
            'caption_to_image_loss': contrastive_losses['caption_to_image_loss'],
            'image_to_entity_loss': contrastive_losses['image_to_entity_loss'],
            'entity_to_image_loss': contrastive_losses['entity_to_image_loss']
        })

        if self.image_caption_hardest_loss_weight > 0:
            image_caption_hardest_loss = image_caption_triplet_loss(
                all_img_embeds, all_cap_embeds, margin=self.hardest_margin, img_indices=img_indices
            )
            total_loss = total_loss + self.image_caption_hardest_loss_weight * image_caption_hardest_loss
            out.update({
                'image_caption_hardest_loss': image_caption_hardest_loss
            })

        if self.specificity_loss_weight > 0:
            specificity_loss = triplet_specificity_loss(
                all_img_embeds, all_cap_embeds, all_ent_embeds, all_ent_img_ids, margin=self.specificity_margin
            )
            total_loss = total_loss + specificity_loss * self.specificity_loss_weight
            out.update({
                'specificity_loss': specificity_loss,
            })

        out.update({
            'total_loss': total_loss,
        })

        return out

    def prepare_image_entity_mask(self, img_entity_pos_mask):
        """
        Args:
        * img_entity_pos_mask (n_images, MAX): mask to show which (image, entity) pair is positive.

        Why do this?
        First, look at collate_fn in image_caption.py to understand how img_entity_pos_mask is constructed.
        After gathering from all gpus, img_entity_pos_mask is a stack of (n_images, MAX) from each gpu. Each gpu can
        have different number of entities, hence different number of columns with full -1. The goal of this function
        is to remove these full -1 columns.
        """
        masks = []
        n_entities = []

        for i in range(self.n_gpus):
            fr = i * self.batch_per_gpu
            to = fr + self.batch_per_gpu

            # find the pos_mask from gpu i
            M = img_entity_pos_mask[fr:to]
            masks.append(M)

            # Since we set an entire column to -1, we can check M against -batch_per_gpu to see which are real
            # entities and which are padded (see collate_fn for more details).
            valid_mask = (M.sum(dim=0) != -self.batch_per_gpu)

            # record number of entities on gpu i
            n_entities.append(valid_mask.sum())
        
        out_mask = torch.zeros(
            (img_entity_pos_mask.shape[0], sum(n_entities)), device=img_entity_pos_mask.device
        ).bool()
        n = 0
        for i in range(self.n_gpus):
            fr = i * self.batch_per_gpu
            to = fr + self.batch_per_gpu
            out_mask[fr:to,n:n+n_entities[i]] = masks[i][:,:n_entities[i]]
            n += n_entities[i]

        return out_mask

    def is_frozen_bert_layer(self, n):
        frozen_layers = [
            'layer.0.', 'layer.1.', 'layer.2.', 'layer.3.',
            'layer.4.',
            'layer.5.',
            'layer.6.',
            'layer.7.',
            'layer.8.',
            'layer.9.',
            'layer.10.',
            'layer.11.',
            '.embeddings.',
        ][:self.freeze_bert_until]
        # frozen_layers.append('.embeddings.')
        return ('bert_embedding' in n or 'bert_prefix_model' in n) and any([x in n for x in frozen_layers])
        
    def freeze_bert(self):
        for n, p in self.named_parameters():
            if 'bert_embedding' in n and self.is_frozen_bert_layer(n):
                p.requires_grad = False

    def freeze_bert_prefix(self):
        for n, p in self.named_parameters():
            if 'bert_prefix_model' in n and self.is_frozen_bert_layer(n):
                p.requires_grad = False

    def freeze_bert_all(self):
        for n, p in self.named_parameters():
            if 'bert_embedding' in n:
                p.requires_grad = False

    def unfreeze_bert_all(self):
        for n, p in self.named_parameters():
            if 'bert_embedding' in n:
                p.requires_grad = True

    def create_optimizer(self, cfg):
        if cfg.MODEL.freeze_bert:
            self.freeze_bert()
            self.freeze_bert_prefix()

        image_encoder_params = [
            p for n, p in self.image_encoder.named_parameters() if p.requires_grad
        ]
        entity_composer_params = [
            p for n, p in self.entity_composer.named_parameters() if p.requires_grad and 'word_embeddings' not in n and 'bert_embedding' not in n and 'bert_prefix_model' not in n
        ]
        relation_composer_params = [
            p for n, p in self.relation_composer.named_parameters() if p.requires_grad and 'word_embeddings' not in n and 'bert_embedding' not in n and 'bert_prefix_model' not in n
        ]
        word_params = [
            p for n, p in self.entity_composer.named_parameters() if p.requires_grad and 'word_embeddings' in n and 'bert_embedding' not in n and 'bert_prefix_model' not in n
        ]
        bert_params = [
            p for n, p in self.entity_composer.named_parameters() if p.requires_grad and 'bert_embedding' in n
        ]
        image_encoder_backbone_params = [
            p for n, p in self.image_encoder.named_parameters() if p.requires_grad and 'backbone_cnn' in n
        ]
        bert_prefix_params = [
            p for n, p in self.entity_composer.named_parameters() if p.requires_grad and 'bert_prefix_model' in n and 'prefix_encoder' in n
        ]
        bert_prefix_bertemb_params = [
            p for n, p in self.entity_composer.named_parameters() if p.requires_grad and 'bert_prefix_model' in n and ('prefix_encoder' not in n)
        ]
        
        params_group = [
            { "params": image_encoder_params, "lr": cfg.TRAIN.lr_img },
            { "params": entity_composer_params, "lr": cfg.TRAIN.lr },
            { "params": relation_composer_params, "lr": cfg.TRAIN.lr },
            { "params": word_params, "lr": cfg.TRAIN.lr_word },
        ]
        if len(bert_params) > 0:
            params_group += [{ "params": bert_params, "lr": cfg.TRAIN.lr * 0.1, "weight_decay": 0.01 }]
        if len(image_encoder_backbone_params) > 0:
            params_group += [{ "params": image_encoder_backbone_params, "lr": cfg.TRAIN.lr * 0.01 }]
        if len(bert_prefix_params) > 0:
            params_group += [{ "params": bert_prefix_params, "lr": cfg.TRAIN.lr }]
        if len(bert_prefix_bertemb_params) > 0:
            params_group += [{ "params": bert_prefix_bertemb_params, "lr": cfg.TRAIN.lr * 0.1, "weight_decay": 0.01 }]

        for i, group in enumerate(params_group):
            print(f"Group {i}: {len(group['params'])}")
        
        self.all_params = image_encoder_params + entity_composer_params + relation_composer_params + word_params + bert_params + image_encoder_backbone_params + bert_prefix_params + bert_prefix_bertemb_params
        print("Trainable params", sum([p.numel() for p in self.all_params]))
        optimizer = optim.AdamW(params_group, lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay)
        return optimizer, params_group

    def should_decay_learning_rate(self, epoch, cfg):
        lr_decay_epochs = cfg.TRAIN.lr_decay_epochs
        if epoch in lr_decay_epochs:
            if epoch in lr_decay_epochs:
                print(f'Reach epoch {epoch} --> decay lr')
                lr_decay_epochs.remove(epoch)
            return True
        return False

    def adjust_learning_rate(self, optimizer, cfg):
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = param_group["lr"]
            new_lr = old_lr * cfg.TRAIN.lr_decay_rate
            param_group["lr"] = new_lr
            print(f'Group {i}:', param_group["lr"])


def build_cora_model(cfg, vocab_size=-1):
    # Build image encoder
    image_encoder = build_image_encoder(cfg)

    # Build entity composer (compose objects and attributes to create entities)
    entity_composer = build_entity_composer(cfg, vocab_size=vocab_size)

    # Build relation composer (composer entities and relations)
    relation_composer = build_relation_composer(
        cfg, vocab_size=vocab_size
    )
    
    if entity_composer.composer_method not in ['bert', 'phrase-bert', 'bert-prefix']:
        relation_composer.set_word_embeddings_weight(entity_composer.word_embeddings.weight)
    else:
        if entity_composer.composer_method == 'bert-prefix':
            relation_composer.set_bert_model(entity_composer.bert_prefix_model)
        else:
            relation_composer.set_bert_model(entity_composer.bert_embedding)

    model = CoraModel(
        image_encoder,
        entity_composer,
        relation_composer,
    )
    return model