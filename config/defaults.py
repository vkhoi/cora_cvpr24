from yacs.config import CfgNode as CN


_C = CN()
_C.config_name = 'cora_gru'

# -----------------------------------------------------------------------
# Distributed
# -----------------------------------------------------------------------
_C.DISTRIBUTED = CN()
_C.DISTRIBUTED.backend = 'nccl'
_C.DISTRIBUTED.world_size = 4
_C.DISTRIBUTED.port = 1427

# -----------------------------------------------------------------------
# Wandb
# -----------------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.project = 'cora'
_C.WANDB.entity = 'vkhoi'
_C.WANDB.use = False
_C.WANDB.id = ''

# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
_C.DATASET = CN(new_allowed=True)
_C.DATASET.directory = '/fs/vast-cfar-projects/cora/f30k'

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------
_C.MODEL = CN(new_allowed=True)

_C.MODEL.img_dropout = 0.1
_C.MODEL.img_encoder_type = 'region'
_C.MODEL.img_pooling_type = 'gpo-selfattn'      # 'avg' or 'gpo-selfattn'
_C.MODEL.img_pos_embed_dim = 16

_C.MODEL.word_embed_source = 'scratch'
_C.MODEL.word_embed_dim = 300

_C.MODEL.entity_composer_method = 'gru'
_C.MODEL.entity_composer_hidden_dim = 1024
_C.MODEL.entity_composer_n_layers = 1
_C.MODEL.entity_composer_dropout = 0.15
_C.MODEL.entity_bert_pooling = 'avg'            # only used when composer is BERT

_C.MODEL.rel_composer_method = 'gru'
_C.MODEL.rel_composer_hidden_dim = 1024
_C.MODEL.rel_composer_n_layers = 1
_C.MODEL.rel_composer_dropout = 0.15
_C.MODEL.rel_bert_pooling = 'avg'               # only used when composer is BERT

_C.MODEL.graph_pos_embed_dim = 16
_C.MODEL.use_graph_global = False

_C.MODEL.embed_dim = 1024

_C.MODEL.freeze_bert = False

_C.MODEL.weights = ''
_C.MODEL.optim_weights = ''
_C.MODEL.pretrained_word_embedding_path = ''

# -----------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------
_C.TRAIN = CN(new_allowed=True)

_C.TRAIN.checkpoint_directory = '/fs/nexus-scratch/khoi/cora/checkpoints/f30k'
_C.TRAIN.seed = 124

_C.TRAIN.num_workers = 4

_C.TRAIN.batch_size = 128
_C.TRAIN.start_epoch = 0
_C.TRAIN.start_evaluate_epoch = 1
_C.TRAIN.num_epochs = 55

_C.TRAIN.contrastive_temperature = 0.01

_C.TRAIN.specificity_loss_weight = 11.0
_C.TRAIN.specificity_margin = 0.35

_C.TRAIN.image_caption_hardest_loss_weight = 4.0
_C.TRAIN.hardest_margin = 0.35

_C.TRAIN.lr_img = 5e-4                          # lr for image encoder
_C.TRAIN.lr_word = 5e-4                         # lr for word embedding
_C.TRAIN.lr = 5e-4                              # lr for other modules
_C.TRAIN.lr_warmup_epochs = 1                   # num of epochs to warm up
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.clip_grad = -1.0                       # if not apply then set to -1
_C.TRAIN.lr_decay_epochs = [15]                 # epoch to decay lr
_C.TRAIN.lr_decay_rate = 0.1                    # decay factor

_C.TRAIN.region_dropout = True
_C.TRAIN.region_dropout_prob1 = 1.0
_C.TRAIN.region_dropout_prob2 = 0.35
_C.TRAIN.grid_dropout = 0.2

_C.TRAIN.max_n_entities = 13
_C.TRAIN.max_n_entity_tokens = 8
_C.TRAIN.max_n_attributes = 4
_C.TRAIN.max_n_attribute_tokens = 5
_C.TRAIN.max_n_relations = 13
_C.TRAIN.max_n_relation_tokens = 9

_C.TRAIN.prob_drop_entity = 0.05
_C.TRAIN.prob_drop_attribute = 0.15
_C.TRAIN.prob_drop_relation = 0.15

_C.TRAIN.prob_mask_entity_token = 0.10
_C.TRAIN.prob_delete_entity_token = 0.05
_C.TRAIN.prob_replace_entity_token = 0.0

_C.TRAIN.prob_mask_relation_token = 0.10
_C.TRAIN.prob_delete_relation_token = 0.05
_C.TRAIN.prob_replace_relation_token = 0.0

_C.TRAIN.prob_singularize_obj = 0.125
_C.TRAIN.prob_pluralize_obj = 0.10

_C.TRAIN.disp_interval = 300
_C.TRAIN.eval_every_epoch = 1

# -----------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------
_C.TEST = CN(new_allowed=True)