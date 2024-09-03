def record_hyperparams(cfg):
    hyperparams = {
        "MODEL.img_dropout": cfg.MODEL.img_dropout,
        "MODEL.img_encoder_type": cfg.MODEL.img_encoder_type,
        "MODEL.img_pooling_type": cfg.MODEL.img_pooling_type,

        "MODEL.entity_composer_method": cfg.MODEL.entity_composer_method,
        "MODEL.entity_composer_hidden_dim": cfg.MODEL.entity_composer_hidden_dim,
        "MODEL.entity_composer_n_layers": cfg.MODEL.entity_composer_n_layers,
        "MODEL.entity_composer_dropout": cfg.MODEL.entity_composer_dropout,

        "MODEL.rel_composer_method": cfg.MODEL.rel_composer_method,
        "MODEL.rel_composer_hidden_dim": cfg.MODEL.rel_composer_hidden_dim,
        "MODEL.rel_composer_n_layers": cfg.MODEL.rel_composer_n_layers,
        "MODEL.rel_composer_dropout": cfg.MODEL.rel_composer_dropout,

        "MODEL.embed_dim": cfg.MODEL.embed_dim,

        "MODEL.use_graph_global": cfg.MODEL.use_graph_global,

        "MODEL.weights": cfg.MODEL.weights,
        "MODEL.pretrained_word_embedding_path": cfg.MODEL.pretrained_word_embedding_path,

        ###############

        "TRAIN.seed": cfg.TRAIN.seed,
        "TRAIN.batch_size": cfg.TRAIN.batch_size,
        "TRAIN.lr": cfg.TRAIN.lr,
        "TRAIN.lr_img": cfg.TRAIN.lr_img,
        "TRAIN.lr_word": cfg.TRAIN.lr_word,
        "TRAIN.weight_decay": cfg.TRAIN.weight_decay,
        "TRAIN.clip_grad": cfg.TRAIN.clip_grad,
        "TRAIN.lr_decay_epochs": cfg.TRAIN.lr_decay_epochs,
        "TRAIN.lr_decay_rate": cfg.TRAIN.lr_decay_rate,

        "TRAIN.contrastive_temperature": cfg.TRAIN.contrastive_temperature,
        "TRAIN.specificity_loss_weight": cfg.TRAIN.specificity_loss_weight,
        "TRAIN.specificity_margin": cfg.TRAIN.specificity_margin,
        "TRAIN.image_caption_hardest_loss_weight": cfg.TRAIN.image_caption_hardest_loss_weight,
        "TRAIN.hardest_margin": cfg.TRAIN.hardest_margin,

        "TRAIN.region_dropout_prob1": cfg.TRAIN.region_dropout_prob1,
        "TRAIN.region_dropout_prob2": cfg.TRAIN.region_dropout_prob2,

        "TRAIN.max_n_entities": cfg.TRAIN.max_n_entities,
        "TRAIN.max_n_attributes": cfg.TRAIN.max_n_attributes,
        "TRAIN.max_n_relations": cfg.TRAIN.max_n_relations,
        "TRAIN.max_n_entity_tokens": cfg.TRAIN.max_n_entity_tokens,
        "TRAIN.max_n_attribute_tokens": cfg.TRAIN.max_n_attribute_tokens,
        "TRAIN.max_n_relation_tokens": cfg.TRAIN.max_n_relation_tokens,

        "TRAIN.prob_singularize_obj": cfg.TRAIN.prob_singularize_obj,
        "TRAIN.prob_pluralize_obj": cfg.TRAIN.prob_pluralize_obj,
        "TRAIN.prob_drop_entity": cfg.TRAIN.prob_drop_entity,
        "TRAIN.prob_drop_attribute": cfg.TRAIN.prob_drop_attribute,
        "TRAIN.prob_drop_relation": cfg.TRAIN.prob_drop_relation,

        "TRAIN.prob_mask_entity_token": cfg.TRAIN.prob_mask_entity_token,
        "TRAIN.prob_delete_entity_token": cfg.TRAIN.prob_delete_entity_token,
        "TRAIN.prob_replace_entity_token": cfg.TRAIN.prob_replace_entity_token,
        "TRAIN.prob_mask_relation_token": cfg.TRAIN.prob_mask_relation_token,
        "TRAIN.prob_delete_relation_token": cfg.TRAIN.prob_delete_relation_token,
        "TRAIN.prob_replace_relation_token": cfg.TRAIN.prob_replace_relation_token,
    }
    return hyperparams
