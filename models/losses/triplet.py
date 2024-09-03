import torch
import torch.nn as nn


def triplet_specificity_loss(img, cap, ent, ent_img_ids, margin=0.2):
    """Computs triplet loss to enfore specificity between image and caption+entity.

    We want to encourage the model to produce higher score for caption than for entity, since caption is more specific.
    
    Args:
    * img: image embeddings (n_images, d)
    * cap: caption embeddings (n_captions, d)
    * ent: image embeddings (n_entities, d)
    * ent_img_ids: image id of each entity (n_entities,) - to show which image an entity belongs to.
    """
    n_images = img.shape[0]
    n_entities = ent.shape[0]

    scores_img_to_ent = img @ ent.t() # (n_images, n_entities)
    scores_img_to_cap = (img * cap).sum(1, keepdim=True) # (n, 1)

    # mask to describe which elements in scores_img_to_ent are positive pairs
    gt_entity_mask = torch.zeros((n_entities, n_images), device=img.device).bool()
    gt_entity_mask[torch.arange(n_entities), ent_img_ids] = True
    gt_entity_mask = gt_entity_mask.t() # (n_images, n_entities)

    loss = (scores_img_to_ent + margin - scores_img_to_cap).clamp(min=0)
    loss = loss[gt_entity_mask].mean()

    return loss


def image_caption_triplet_loss(img, cap, margin=0.2, img_indices=None):
    """Computes VSE++ triplet loss between image and caption.
    """
    scores = img @ cap.t()
    diagonal = scores.diag().view(img.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # text retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = (torch.eye(scores.size(0)) > .5)
    if torch.cuda.is_available():
        mask = mask.cuda()
    if img_indices is not None:
        same_img_mask = (img_indices.unsqueeze(0) == img_indices.unsqueeze(1))
        mask = mask | same_img_mask
    cost_s = cost_s.masked_fill_(mask, 0)
    cost_im = cost_im.masked_fill_(mask, 0)

    cost_s = cost_s.max(1)[0]
    cost_im = cost_im.max(0)[0]

    return cost_s.mean() + cost_im.mean()
