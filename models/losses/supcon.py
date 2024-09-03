# Source based on: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
import torch


def supcon_joint_caption_entity_loss(
    img_embeds, cap_embeds, ent_embeds, ent_img_ids, temperature=0.05,
    img_ent_pos_mask=None,
    ent_mask=None,
    img_indices=None,
):
    """Computes supervised contrastive loss of image against caption+entity (and vice versa).

    Here we know the number of images and captions are the same.
    Args:
    * img_embeds: image embeddings (n_images, d)
    * cap_embeds: caption embeddings (n_captions, d)
    * ent_embeds: image embeddings (n_entities, d)
    * ent_img_ids: image id of each entity (n_entities,) - to show which image an entity belongs to.
    * temperature: softmax temperature.
    * img_ent_pos_mask: to show which (image, entity) are positive pairs.
    """
    n_images = img_embeds.shape[0]
    n_entities = ent_embeds.shape[0]

    cap_and_ent_embeds = torch.cat((cap_embeds, ent_embeds), dim=0)
    scores = img_embeds @ cap_and_ent_embeds.t() / temperature # (n_images, n_captions + n_entities)

    # create mask to describe which elements in the scores matrix represent positive pairs
    # we start with an identity matrix
    cap_mask = torch.eye(n_images, device=img_embeds.device)

    # find (image, caption) positive pairs
    if img_indices is not None:
        # There's a chance that an image is sampled twice (or more) in the batch, so we use image index to find
        # these positive pairs.
        same_img_mask = (img_indices.unsqueeze(0) == img_indices.unsqueeze(1)).to(img_embeds.device)
        cap_mask = cap_mask.bool() | same_img_mask

    # find (image, entity) positive pairs
    if ent_mask is None:
        ent_mask = torch.zeros((n_entities, n_images), device=img_embeds.device).bool()
        ent_mask[torch.arange(n_entities), ent_img_ids] = True
        ent_mask = ent_mask.t() # (n_images, n_entities)
    if img_ent_pos_mask is not None:
        ent_mask = ent_mask | img_ent_pos_mask
    mask = torch.cat((cap_mask, ent_mask), dim=1) # (n_images, n_captions + n_entities)

    # minus max value for numerical stability
    logits_max, _ = torch.max(scores, dim=1, keepdim=True)
    logits = scores - logits_max.detach()
    exp_logits = torch.exp(logits)

    # Mask away positive pairs. (SupCon paper does this but their PyTorch implementation doesn't).
    # Source: https://github.com/HobbitLong/SupContrast/issues/64.
    exp_logits_masked_positive = exp_logits * (1 - mask.float())
    exp_logits_masked_positive_sum = exp_logits_masked_positive.sum(1, keepdim=True) # (n_images, 1)
    
    # this is the denominator in the contrastive loss
    exp_logits_masked_positive_sum = exp_logits_masked_positive_sum + exp_logits + 1e-9

    # compute log likelihood
    log_prob = logits - torch.log(exp_logits_masked_positive_sum)

    # we aggregate log prob of all positive pairs to align the positive pair embeddings close together
    mask_log_prob = mask * log_prob

    mask_sum = mask.sum(1) # number of positive pairs per image
    mean_log_prob_pos = mask_log_prob.sum(1) / mask_sum # mean loss per image
    loss1 = -mean_log_prob_pos.mean() # mean image-to-caption&entity loss

    # Next, break down image to caption loss and image to entity loss (for loss plotting and understanding purpose).
    # image to caption
    image_to_caption_loss = -(mask_log_prob[:,:n_images].sum(1) / mask[:,:n_images].sum(1)).mean()
    # image to entity
    image_to_entity_loss = mask_log_prob[:,n_images:].sum(1)
    num_entities_per_image = mask[:,n_images:].sum(1)
    non_empty_entities_mask = num_entities_per_image > 0
    image_to_entity_loss = -(
        image_to_entity_loss[non_empty_entities_mask] / (num_entities_per_image[non_empty_entities_mask] + 1e-9)
    ).mean()

    out = {
        'image_to_caption_loss': image_to_caption_loss,
        'image_to_entity_loss': image_to_entity_loss,
    }

    # do the same as above, but now this is image retrieval using caption and entities as inputs
    scores = scores.t()
    mask = mask.t()
    logits_max, _ = torch.max(scores, dim=1, keepdim=True)
    logits = scores - logits_max.detach()
    exp_logits = torch.exp(logits)

    exp_logits_masked_positive = exp_logits * (1 - mask.float())
    exp_logits_masked_positive_sum = exp_logits_masked_positive.sum(1, keepdim=True) # (n_captions + n_entities, 1)
    exp_logits_masked_positive_sum = exp_logits_masked_positive_sum + exp_logits + 1e-6
    log_prob = logits - torch.log(exp_logits_masked_positive_sum)

    mask_log_prob = mask * log_prob
    
    mask_sum = mask.sum(1) # number of positive images per caption and entity
    mean_log_prob_pos = mask_log_prob.sum(1) / mask_sum
    loss2 = -mean_log_prob_pos.mean()

    caption_to_image_loss = -mean_log_prob_pos[:n_images].mean()
    entity_to_image_loss = -mean_log_prob_pos[n_images:].mean()
    
    out.update({
        'total_loss': loss1 + loss2,
        'caption_to_image_loss': caption_to_image_loss,
        'entity_to_image_loss': entity_to_image_loss
    })

    return out
