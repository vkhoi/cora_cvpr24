import argparse
import glob
import json
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.utils.clip_grad import clip_grad_norm_

from config import cfg

from datasets.image_caption import BaseDataset, ImageCaptionDataset
from datasets.image_caption import collate_fn as my_collate_fn

from utils.average_meter import AverageMeter
from utils.count_parameters import count_parameters
from utils.evaluation import i2t, t2i
from utils.record_hyperparams import record_hyperparams

import models.cora_model as cora_model


def validate(val_loader, model, device, cfg, split='test'):
    print(f'Evaluating on {split} split')
    model.eval()

    img_embeds = []
    cap_embeds = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            for k in batch:
                if isinstance(batch[k], list):
                    continue
                batch[k] = batch[k].to(device, non_blocking=True)

            out = model(batch)
            
            img_embed = out['img_embeds']
            cap_embed = out['cap_embeds']
            img_embeds.append(img_embed.cpu())
            cap_embeds.append(cap_embed.cpu())

            if device == 'cuda:0':
                if (i+1) % 50 == 0:
                    print(f'Done {i+1}/{len(val_loader)} batches', flush=True)

    if split == 'testall':
        img_embeds = torch.cat(img_embeds, dim=0)
        cap_embeds = torch.cat(cap_embeds, dim=0)

        results = []
        for i in range(5):
            img_embs_shard = img_embeds[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embeds[i * 5000:(i + 1) * 5000]
            start = time.time()
            sims = torch.matmul(img_embs_shard, cap_embs_shard.t())
            end = time.time()
            print("calculate similarity time: {}".format(end - start))
            print(sims.shape, flush=True)
            sims = sims.numpy()
            npts = sims.shape[0]
            (r1, r5, r10, medr, meanr) = i2t(npts, sims)
            (r1i, r5i, r10i, medri, meanri) = t2i(npts, sims)

            ar = (r1 + r5 + r10) / 3
            ari = (r1i + r5i + r10i) / 3
            rsum = r1 + r5 + r10 + r1i + r5i + r10i
            results += [[r1, r5, r10, medr, meanr] + [r1i, r5i, r10i, medri, meanri] + [ar, ari, rsum]]
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print(f'rsum: {mean_metrics[12]:.1f}')
        print(f'Avg i2t: {mean_metrics[10]:.1f}')
        print(f'Image to text: {mean_metrics[0]:.1f} {mean_metrics[1]:.1f} {mean_metrics[2]:.1f} {mean_metrics[3]:.1f} {mean_metrics[4]:.1f}')
        print(f'Avg t2i: {mean_metrics[11]:.1f}')
        print(f'Text to image: {mean_metrics[5]:.1f} {mean_metrics[6]:.1f} {mean_metrics[7]:.1f} {mean_metrics[8]:.1f} {mean_metrics[9]:.1f}')
        return mean_metrics[12]
    else:
        img_embeds = torch.cat(img_embeds, dim=0)
        img_embeds = torch.cat([img_embeds[i].unsqueeze(0) for i in range(0, len(img_embeds), 5)])
        cap_embeds = torch.cat(cap_embeds, dim=0)

        start = time.time()
        sims = torch.matmul(img_embeds, cap_embeds.t())
        end = time.time()
        if device == 'cuda:0':
            print("calculate similarity time: {}".format(end - start))

        sims = sims.numpy()

        # caption retrieval
        npts = sims.shape[0]
        # (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
        (r1, r5, r10, medr, meanr) = i2t(npts, sims)
        if device == 'cuda:0':
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1, r5, r10, medr, meanr), flush=True)

        # image retrieval
        # (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
        (r1i, r5i, r10i, medri, meanri) = t2i(npts, sims)
        if device == 'cuda:0':
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1i, r5i, r10i, medri, meanri), flush=True)
        # sum of recalls to be used for early stopping
        currscore = r1 + r5 + r10 + r1i + r5i + r10i
        if device == 'cuda:0':
            print('Current rsum is {}'.format(currscore), flush=True)

        return currscore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--split', type=str, required=True,
                        help="'dev' or 'test'")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    print(cfg)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    torch.cuda.set_device(0)
    device = 'cuda:0'

    # Prepare dataset & dataloader.
    print('Prepare dataset', flush=True)
    base_dataset = BaseDataset(cfg, split=args.split)
    dataset = ImageCaptionDataset(base_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    if cfg.MODEL.word_embed_source not in ['bert', 'bert-prefix']:
        # if not using BERT as text encoder, we need to initialize our own vocab embedding matrix
        vocab_size = len(base_dataset.vocab.word2idx)
    else:
        vocab_size = -1
    
    print('Prepare model', flush=True)
    model = cora_model.build_cora_model(cfg, vocab_size=vocab_size)
    model.setup_training_loss(cfg)
    model.to(device)

    if cfg.MODEL.weights != '':
        print('Load weights from checkpoint at %s' % cfg.MODEL.weights)
        weights_dict = torch.load(cfg.MODEL.weights, map_location=device)
        model_state_dict = model.state_dict()
        unloaded_dict = {
            k:v for k,v in weights_dict.items()
            if (k not in model_state_dict) or (v.shape != model_state_dict[k].shape) }
        weights_dict = {
            k: v for k,v in weights_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape}
        missing_dict = {
            k: v for k,v in model_state_dict.items()
            if (k not in weights_dict) or (v.shape != weights_dict[k].shape) }
        if device == 'cuda:0':
            print('-----------------------------')
            print('Unable to load the following weights')
            print(unloaded_dict.keys())
            print('Missing the following weights')
            print(missing_dict.keys())
        model.load_state_dict(weights_dict, strict=True)

    torch.backends.cudnn.benchmark = True

    print('Start evaluating', flush=True)

    score = validate(dataloader, model, device, cfg, split=args.split)


if __name__ == "__main__":
    main()
