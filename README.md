# Composing Object Relations and Attributes for Image-Text Matching
Implementation of the paper [Composing Object Relations and Attributes for Image-Text Matching](https://arxiv.org/abs/2406.11820) (CVPR 2024).

Please cite our paper using the following bib entry if you find it useful for your work.
```
@inproceedings{pham2024composing,
  title={Composing object relations and attributes for image-text matching},
  author={Pham, Khoi and Huynh, Chuong and Lim, Ser-Nam and Shrivastava, Abhinav},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14354--14363},
  year={2024}
}
```

### Notes
* We have released most of our codebase. The code is now usable for training our model using GRU as the text encoder.
* TODO: release the code and config to train our model wih GloVe embedding and BERT text encoder on both Flick30K and COCO dataset.
* TODO: release the pretrained checkpoints.
* TODO: release the code for the scene graph parser. In the meantime, you can checkout a great work on textual scene graph parsing: [FACTUAL](https://github.com/zhuang-li/FactualSceneGraph).

## Prerequisites
### Environment
The experiments in this project were done with the following key dependencies:
* Python 3.10.4
* PyTorch 1.13.0
* cuda 11.7.0

```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

To use tokenizer from nltk, `punkt_tab` is also needed.
```
python
>>> import nltk
>>> nltk.download('punkt_tab')
```

### Data
Download the data from [Link](https://drive.google.com/drive/folders/1fCrRPAN_FtdZUjHI6gaW3csC1lthdkiu?usp=sharing), then extract it and make sure it is in the following structure. Structure of `f30k` and `coco` should be similar.

```
data
├── f30k
│   ├── scene_graph # extracted scene graphs for each split
│   │      ├── train.json
│   │      ├── dev.json
│   │      ├── test.json
│   │
│   ├── images # f30k images
│   │      ├── xxx.jpg
│   │      └── ...
│   │
│   ├── region_feats
│   │      ├── train            # stores region features of train split
│   │      ├── dev              # stores region features of dev split
│   │      └── test             # stores region features of test split
│   │
│   ├── vocab # data for the vocabulary
│   │      ├── vocab.json       # maps from token to index (and vice versa)
│   │      └── tok2lemma.json   # maps from token to its lemma (e.g., 'cars' -> 'car')
│   │
│   ├── *_caps.txt              # captions of 'train', 'dev', 'test' split
│   ├── *_ids.txt               # image ids of 'train', 'dev', 'test' split
│   └── id_mappings.json        # maps from image id to image filename
│
├── coco
│   ├──....
│   ...
```

For the region features, instead of using a single file as in previous work (e.g., as downloaded from [here](https://www.kaggle.com/datasets/kuanghueilee/scan-features)), we split the region features into multiple smaller files and only read the corresponding region feature file when necessary from the dataloader (refer to `notebooks/split_regions.ipynb` for details on how it is splitted). This allows us not having to load all region features into memory so that we can use a larger number of workers for the dataloader. Note that the images are not necessary for the code to work when running in 'region features' mode.

The scene graph parser code will be released. In the meantime, please checkout the following great work on parsing textual scene graphs: [FACTUAL](https://github.com/zhuang-li/FactualSceneGraph). With some modifications, the scene graphs produced from this work can also be used within our code.

## Pretrained checkpoints
All pretrained checkpoints will be uploaded [here](https://drive.google.com/drive/folders/1CcFFYMzY8eu7YDbM8YNQNqAMCjcFSFnN?usp=drive_link).
* `f30k_gru_scratch.pth`: on Flickr30K, GRU, trained from scratch.

## Training
All config files can be found in the `config` folder. To run training, find the config file corresponding to the settings you want to run, make adjustment for your environment, hyperparams, etc., and launch training with
```
# Flickr30K, GRU as text encoder, train with 4 GPUs
python -m main.train --cfg config/f30/gru.yaml \
    DISTRIBUTED.world_size 4 TRAIN.num_workers 12
```

## Testing
```
# Flickr30K, GRU as text encoder, evaluate on test split
python -m --cfg config/f30k/gru.yaml --split test \
    MODEL.weights <directory to the checkpoint file>
```