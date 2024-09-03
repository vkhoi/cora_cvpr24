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
import wandb

from torch.nn.utils.clip_grad import clip_grad_norm_

from config import cfg

from datasets.image_caption import BaseDataset, ImageCaptionDataset
from datasets.image_caption import collate_fn as my_collate_fn

from utils.average_meter import AverageMeter
from utils.count_parameters import count_parameters
from utils.evaluation import i2t, t2i
from utils.record_hyperparams import record_hyperparams

import models.cora_model as cora_model


def train(epoch, train_loader, model, optimizer, device, cfg):
    print(f'Training: Epoch {epoch+1}')
    model.train()

    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    end_time = time.time()

    # use these meters to keep track of the average values of some metrics
    list_meters = [
        'total_loss',
        'image_to_caption_loss',        # image-to-text contrastive loss
        'caption_to_image_loss',        # text-to-image contrastive loss
        'image_to_entity_loss',         # image-to-entity contrastive loss
        'entity_to_image_loss',         # entity-to-image contrastive loss
        'specificity_loss',
        'image_caption_hardest_loss',   # vse++ loss
        'n_regions_after_dropout',      # after dropout input regions, how many regions remain
    ]
    dict_meters = { 
        k: AverageMeter() for k in list_meters
    }

    # number of past training iterations 
    n_past_iters = epoch * len(train_loader)

    # number of iterations to warm up learning rate
    n_lr_warmup_iters = cfg.TRAIN.lr_warmup_epochs * len(train_loader)

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end_time)
        for k in batch:
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(device, non_blocking=True)

        out = model(batch, epoch=epoch)
        if n_past_iters+i+1 <= n_lr_warmup_iters:
            warmup_loss = (n_past_iters+i+1) / n_lr_warmup_iters * out['total_loss']
            out['total_loss'] = warmup_loss

        for param in model.parameters():
            param.grad = None
        out['total_loss'].backward()
        if cfg.TRAIN.clip_grad > 0.0:
            if isinstance(model, nn.parallel.DistributedDataParallel): 
                clip_grad_norm_(model.module.all_params, cfg.TRAIN.clip_grad)
            else:
                clip_grad_norm_(model.all_params, cfg.TRAIN.clip_grad)
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # keep track of average values of all metrics
        for k in out:
            if k in dict_meters:
                dict_meters[k].update(out[k].item())

        if ((n_past_iters+i+1) % cfg.TRAIN.disp_interval == 0) or ((n_past_iters+i+1) == 50):
            if device == 'cuda:0':
                # only log output on process 0
                if cfg.WANDB.use:
                    for k in dict_meters:
                        if dict_meters[k].avg > 0.0: # trick to check if a loss is really computed
                            prefix = 'train'
                            wandb.log({
                                f'{prefix}/{k}': dict_meters[k].avg
                            }, commit=False)

                max_gpu_usage_mb = torch.cuda.max_memory_allocated(device=device) / 1048576.0

                print_str = \
                    f"[Epoch {epoch+1}] [Iter {i+1}/{len(train_loader)}] Loss={dict_meters['total_loss'].avg:.6f}"
                print_str += f', Batch_time: {batch_time.avg:.3f}, Data_time: {data_time.avg:.3f}'
                print_str += f', Mem: {max_gpu_usage_mb} MB'
                print(print_str, flush=True)

            for k in dict_meters:
                dict_meters[k].reset()
            batch_time.reset()
            data_time.reset()

            if device == 'cuda:0':
                if cfg.WANDB.use:
                    wandb.log({
                        'train/iter': n_past_iters+i+1
                    }, commit=True)


def compute_scores(sims, write_wandb=False, prefix=''):
    # caption retrieval
    npts = sims.shape[0]
    # (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
        (r1, r5, r10, medr, meanr), flush=True)

    # image retrieval
    # (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    (r1i, r5i, r10i, medri, meanri) = t2i(npts, sims)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
        (r1i, r5i, r10i, medri, meanri), flush=True)
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    print('Current rsum is {}'.format(currscore), flush=True)
    if write_wandb:
        wandb.log({
            f'{prefix}/i2t-r1': r1,
            f'{prefix}/i2t-r5': r5,
            f'{prefix}/i2t-r10': r10,
            f'{prefix}/i2t-medr': medr,
            f'{prefix}/i2t-meanr': meanr,
            #################
            f'{prefix}/t2i-r1': r1i,
            f'{prefix}/t2i-r5': r5i,
            f'{prefix}/t2i-r10': r10i,
            f'{prefix}/t2i-medr': medri,
            f'{prefix}/t2i-meanr': meanri,
            #################
            f'{prefix}/rsum': currscore,
        }, commit=False)

    return currscore


def validate(epoch, val_loader, model, device, cfg, split='dev'):
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
        if cfg.WANDB.use:
            wandb.log({
                f'{split}/i2t-r1': mean_metrics[0],
                f'{split}/i2t-r5': mean_metrics[1],
                f'{split}/i2t-r10': mean_metrics[2],
                f'{split}/i2t-medr': mean_metrics[3],
                f'{split}/i2t-meanr': mean_metrics[4],
                #################
                f'{split}/t2i-r1': mean_metrics[5],
                f'{split}/t2i-r5': mean_metrics[6],
                f'{split}/t2i-r10': mean_metrics[7],
                f'{split}/t2i-medr': mean_metrics[8],
                f'{split}/t2i-meanr': mean_metrics[9],
                #################
                f'{split}/rsum': mean_metrics[12],
            }, commit=False)
        return mean_metrics[12]
    else:
        img_embeds = torch.cat(img_embeds, dim=0)
        img_embeds = torch.cat([img_embeds[i].unsqueeze(0) for i in range(0, len(img_embeds), 5)])
        cap_embeds = torch.cat(cap_embeds, dim=0)

        start = time.time()
        sims = torch.matmul(img_embeds, cap_embeds.t())
        end = time.time()
        print("calculate similarity time: {}".format(end - start))

        scores = compute_scores(sims.numpy(), write_wandb=cfg.WANDB.use, prefix=split)

        return scores


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_checkpoint(model_or_optim, name, cfg, seed):
    """Save checkpoint.
    """
    if isinstance(model_or_optim, nn.parallel.DistributedDataParallel):
        state_dict = model_or_optim.module.state_dict()
    else:
        state_dict = model_or_optim.state_dict()
    path = os.path.join(
        f'{cfg.TRAIN.checkpoint_directory}/{cfg.config_name}_{seed}/{name}.pth')
    torch.save(state_dict, path)


def main_worker(gpu, cfg):
    """Main training code.
    """
    seed = cfg.TRAIN.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if gpu == 0:
        # only initialize wandb on process 0
        if cfg.WANDB.use:
            if cfg.WANDB.id != '':
                # resume existing wandb session
                print(f'Resume wandb session {cfg.WANDB.id}')
                wandb.init(
                    id=cfg.WANDB.id,
                    project=cfg.WANDB.project,
                    resume="must",
                    entity=cfg.WANDB.entity)
            else:
                wandb.init(
                    project=cfg.WANDB.project,
                    name=f"{cfg.config_name}_{cfg.TRAIN.seed}",
                    config=record_hyperparams(cfg),
                    entity=cfg.WANDB.entity)

    print(f'Use GPU {gpu} for training')
    torch.cuda.set_device(gpu)
    device = f'cuda:{gpu}'

    # setup distributed setting
    if cfg.DISTRIBUTED.world_size > 1:
        dist.init_process_group(
            backend=cfg.DISTRIBUTED.backend,
            init_method=f'tcp://127.0.0.1:{cfg.DISTRIBUTED.port}',
            world_size=cfg.DISTRIBUTED.world_size,
            rank=gpu
        )

    # prepare directory for logging and saving checkpoints
    ckpt_path = f'{cfg.TRAIN.checkpoint_directory}/{cfg.config_name}_{seed}'
    if gpu == 0:
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

    # distribute batch size evenly between GPUs
    cfg.TRAIN.batch_size = cfg.TRAIN.batch_size // cfg.DISTRIBUTED.world_size
    print('Batch size on each gpu: %d' % cfg.TRAIN.batch_size)

    # prepare dataset & dataloader
    print('Prepare dataset')
    train_base_dataset = BaseDataset(cfg, split='train')
    train_dataset = ImageCaptionDataset(train_base_dataset, train=True)

    if cfg.DISTRIBUTED.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # ensure same sequence of training examples are drawn from training set
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.batch_size, shuffle=(train_sampler is None),
        num_workers=cfg.TRAIN.num_workers // cfg.DISTRIBUTED.world_size,
        pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=seed_worker, generator=g, collate_fn=my_collate_fn)

    if gpu == 0:
        # only run validation on process 0
        val_base_dataset = BaseDataset(cfg, split='dev')
        val_dataset = ImageCaptionDataset(val_base_dataset)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=64, shuffle=False, pin_memory=True,
            num_workers=cfg.TRAIN.num_workers // cfg.DISTRIBUTED.world_size)

    print('Prepare model')
    if cfg.MODEL.word_embed_source not in ['bert', 'bert-prefix']:
        # if not using BERT as text encoder, we need to initialize our own vocab embedding matrix
        vocab_size = len(train_base_dataset.vocab.word2idx)
    else:
        vocab_size = -1
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
        if device == 'cuda:0':
            print('-----------------------------')
            print('Loaded the following weights')
            print(weights_dict.keys())
            print('-----------------------------')
            print('Unable to load the following weights')
            print(unloaded_dict.keys())
        model.load_state_dict(weights_dict, strict=False)

     # check if training with multiple gpus
    if cfg.DISTRIBUTED.world_size > 1:
        print('Wrap model with DistributedDataParallel')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[gpu],
            find_unused_parameters=True if cfg.MODEL.entity_composer_method in ['bert', 'phrase-bert', 'bert-prefix'] else False)
        optimizer, _ = model.module.create_optimizer(cfg)
    else:
        optimizer, _ = model.create_optimizer(cfg)

    if cfg.MODEL.optim_weights != '':
        print('Load optim weights from checkpoint at %s' % cfg.MODEL.optim_weights)
        weights_dict = torch.load(cfg.MODEL.optim_weights, map_location=device)
        try:
            optimizer.load_state_dict(weights_dict)
        except Exception as e:
            print(e)

    if gpu == 0:
        # only log output on process 0
        table, total_params = count_parameters(model)
        print(table)
        print(f"Total Trainable Params: {total_params}", flush=True)
        
        if cfg.WANDB.use:
            wandb.run.summary["num_parameters"] = total_params
            wandb.define_metric("train/epoch")
            wandb.define_metric("train/iter")
            wandb.define_metric("train/*", step_metric="train/iter")
            wandb.define_metric("dev/*", step_metric="train/epoch")

    torch.backends.cudnn.benchmark = True

    print('Start training', flush=True)
    num_epochs = cfg.TRAIN.num_epochs
    start_epoch = cfg.TRAIN.start_epoch

    best_val = -1

    if isinstance(model, nn.parallel.DistributedDataParallel):
        M = model.module
    else:
        M = model

    for epoch in range(start_epoch, num_epochs):
        if cfg.DISTRIBUTED.world_size > 1:
            train_sampler.set_epoch(epoch)

        train(epoch, train_loader, model, optimizer, device, cfg)

        if gpu == 0 and epoch + 1 >= cfg.TRAIN.start_evaluate_epoch:
            val_score = validate(epoch, val_loader, M, device, cfg)
            if val_score > best_val:
                best_val = val_score
                epoch_best_val = epoch + 1

                save_checkpoint(M, f'model_best', cfg, seed)
                save_checkpoint(optimizer, f'optim_best', cfg, seed)

                if cfg.WANDB.use:
                    wandb.log({
                        'dev/best_rsum': val_score,
                    }, commit=False)
        
        if cfg.DISTRIBUTED.world_size > 1:
            dist.barrier()

        if gpu == 0 and cfg.WANDB.use:
            wandb.log({
                'train/epoch': epoch+1
            }, commit=True)

        if M.should_decay_learning_rate(epoch+1, cfg):
            M.adjust_learning_rate(optimizer, cfg)

    print(best_val, epoch_best_val)

    if gpu == 0:
        if cfg.WANDB.use:
            wandb.finish()

    if cfg.DISTRIBUTED.world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='path to config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # random seed in case seed is not provided in config file
    seed = cfg.TRAIN.seed
    if seed == -1:
        seed = np.random.randint(1, 10000)
    cfg.TRAIN.seed = seed
    print('Random seed:', seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    os.environ["WANDB_START_METHOD"] = "thread" # forgot what this is for

    if cfg.DISTRIBUTED.world_size > 1:
        mp.spawn(main_worker, nprocs=cfg.DISTRIBUTED.world_size, args=(cfg,))
    else:
        main_worker(0, cfg)


if __name__ == "__main__":
    main()
