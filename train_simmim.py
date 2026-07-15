# Copyright (c) Emin Orhan.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import time
import json
import numpy as np
from datetime import timedelta

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

# torchtitan imports
from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_data_loader
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import parallelize_dino, ParallelDims
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

import logging
logging.getLogger("dinov3").setLevel(logging.WARNING)


def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool):
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))
            yield
    return context


def generate_random_mask(batch_size: int, num_patches: int, mask_ratio: float, device: torch.device):
    """Generates a boolean mask of shape (B, L) where True indicates a masked patch."""
    num_masked = int(num_patches * mask_ratio)
    masks = []
    for _ in range(batch_size):
        mask = torch.cat([torch.ones(num_masked), torch.zeros(num_patches - num_masked)])
        mask = mask[torch.randperm(num_patches)]
        masks.append(mask)
    return torch.stack(masks).to(device).bool()


class _BackboneAdapter(nn.Module):
    """Adapter to ensure parallelize_dino.py can find the backbone blocks via the expected attribute path."""
    def __init__(self, backbone):
        super().__init__()
        self.feature_model = backbone


class SimMIM(nn.Module):
    """SimMIM wrapper utilizing DINOv3's native token-space masking."""
    def __init__(self, backbone, encoder_stride, in_chans=3, is_3d=True):
        super().__init__()
        # Wrap backbone to satisfy parallelize_dino expectations 
        self.segmentation_model = nn.ModuleList([_BackboneAdapter(backbone)])
        
        self.encoder_stride = tuple(encoder_stride)
        self.is_3d = is_3d
        self.in_chans = in_chans

        # Extract embed dim directly from the model attribute
        embed_dim = backbone.embed_dim

        # Simple Linear Decoder for SimMIM
        self.patch_dim = in_chans * int(np.prod(self.encoder_stride))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.patch_dim)
        )

    @property
    def backbone(self):
        return self.segmentation_model[0].feature_model

    def forward(self, x, mask=None):
        # By passing is_training=True, DINO returns the full dictionary
        # By passing masks=mask, DINO natively replaces masked locations with self.mask_token
        out = self.backbone(x, masks=mask, is_training=True)
        
        # Grab the dense patch tokens -> Shape: (B, L, embed_dim)
        feats = out["x_norm_patchtokens"]
        
        # Decode to pixel space -> Shape: (B, L, patch_dim)
        pred = self.decoder(feats) 

        if mask is not None:
            # Patchify the original image to serve as the ground truth
            target = self.patchify(x) # Shape: (B, L, patch_dim)
            
            # SimMIM loss: L1 loss computed ONLY on masked patches
            loss = F.l1_loss(pred, target, reduction='none').mean(dim=-1)
            loss = (loss * mask.float()).sum() / (mask.sum() + 1e-5)
            return loss, pred
            
        return pred

    def patchify(self, imgs):
        """Flattens raw images/volumes into patches for direct L1 comparison."""
        B, C = imgs.shape[:2]
        if self.is_3d:
            sD, sH, sW = self.encoder_stride
            D_p, H_p, W_p = imgs.shape[2]//sD, imgs.shape[3]//sH, imgs.shape[4]//sW
            x = imgs.view(B, C, D_p, sD, H_p, sH, W_p, sW)
            x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
            x = x.view(B, C, D_p*H_p*W_p, sD*sH*sW).transpose(1, 2).contiguous()
            return x.view(B, D_p*H_p*W_p, -1)
        else:
            sH, sW = self.encoder_stride
            H_p, W_p = imgs.shape[2]//sH, imgs.shape[3]//sW
            x = imgs.view(B, C, H_p, sH, W_p, sW)
            x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
            x = x.view(B, C, H_p*W_p, sH*sW).transpose(1, 2).contiguous()
            return x.view(B, H_p*W_p, -1)


@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting SimMIM training job: {job_config.job.description}")
    color = utils.Color if job_config.metrics.enable_color_printing else utils.NoColor
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    world_size = int(os.environ['WORLD_SIZE'])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        tp=job_config.training.tensor_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )
    
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)

    gpu_memory_monitor = build_gpu_memory_monitor()
    
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    assert len(job_config.model.crop_size) in (2, 3), "crop_size must have 2 or 3 elements"
    is_3d = len(job_config.model.crop_size) == 3

    # Dataloaders - usually OpenOrganelle Unlabeled Data for self-supervised
    train_loader, _ = build_data_loader(
        job_config.data.dataset_name,
        job_config.data.dataset_path,
        job_config.training.batch_size,
        tuple(job_config.model.crop_size),
        tuple(job_config.model.val_crop_size) if hasattr(job_config.model, 'val_crop_size') else None,
        job_config.data.num_vals,
        job_config.training.seed,
        job_config.training.shuffle_seed,
        dp_rank,
        dp_degree,
        job_config.data.augment
    )

    # build model skeleton (NOTE: we load the pretrained weights during ckpt.load() below)
    backbone = torch.hub.load(
        job_config.model.dinov3_repo_folder, 
        job_config.model.backbone, 
        source="local",
        use_fa4=job_config.model.use_fa4,
        pos_embed_rope_type=job_config.model.rope_type,
        pretrained=False
    )
    
    # Infer encoder stride (assumes 14 if not explicitly provided in config)
    encoder_stride = [16, 16, 16] if is_3d else [16, 16]

    model = SimMIM(
        backbone=backbone,
        encoder_stride=encoder_stride,
        in_chans=3,
        is_3d=is_3d
    )

    if torch.distributed.get_rank() == 0:
        logger.info(f"SimMIM Model Wrapper Initialized: {model}")

    float8_handler = Float8Handler(job_config, parallel_dims)
    float8_handler.convert_to_float8_training(model)

    parallelize_dino(model, world_mesh, parallel_dims, job_config)

    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
    model.to(device=init_device)
    model.train()
    model_parts = [model]

    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

    train_state = TrainState()
    
    checkpoint = CheckpointManager(
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, "Must create seed-checkpoint using one gpu"
        checkpoint.save(curr_step=0, force=True)
        return

    checkpoint.load()

    log_file_handle = None
    if torch.distributed.get_rank() == 0:
        dump_folder = getattr(job_config.job, "dump_folder", ".")
        log_folder = getattr(job_config.metrics, "folder", "logs")
        log_dir = os.path.join(dump_folder, log_folder)
        os.makedirs(log_dir, exist_ok=True)
        log_file_handle = open(os.path.join(log_dir, "metrics.jsonl"), "a")

    train_iterator = iter(train_loader)
    train_context = get_train_context(parallel_dims.loss_parallel_enabled, job_config.experimental.enable_compiled_autograd)

    losses_since_last_log = []
    data_loading_times = []
    time_last_log = time.perf_counter()
    gpu_memory_monitor.reset_peak_stats()
    checkpoint.reset()

    mask_ratio = job_config.model.mask_ratio
    logger.info(f"SimMIM Pre-Training starts at step {train_state.step + 1} with mask ratio {mask_ratio}")

    with maybe_enable_profiling(job_config, global_step=train_state.step) as torch_profiler, \
         maybe_enable_memory_snapshot(job_config, global_step=train_state.step) as memory_profiler:
         
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            data_load_start = time.perf_counter()
            inputs, _ = next(train_iterator) # _ is dummy_label
            data_loading_times.append(time.perf_counter() - data_load_start)
            inputs = inputs.cuda()

            # Calculate L (total number of tokens) for the mask generator
            if is_3d:
                L = (inputs.shape[2] // encoder_stride[0]) * (inputs.shape[3] // encoder_stride[1]) * (inputs.shape[4] // encoder_stride[2])
            else:
                L = (inputs.shape[2] // encoder_stride[0]) * (inputs.shape[3] // encoder_stride[1])

            # Generate random masking pattern (B, L)
            mask = generate_random_mask(inputs.shape[0], L, mask_ratio, inputs.device)

            optimizers.zero_grad()
            
            with train_context():
                loss, preds = model(inputs, mask=mask)
                del preds # Free memory before backward

                loss.backward()
            
            for m in model_parts:
                torch.nn.utils.clip_grad_norm_(m.parameters(), job_config.training.max_norm, foreach=True)

            float8_handler.sync_float8_amax_and_scale_history(model_parts)

            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            losses_since_last_log.append(loss)

            # ====== Log Train Metrics ======
            if (train_state.step == 1 or train_state.step % job_config.metrics.log_freq == 0):
                losses = [l.item() for l in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if parallel_dims.dp_enabled:
                    global_avg_loss = utils.dist_mean(avg_loss, dp_mesh)
                    global_max_loss = utils.dist_max(max_loss, dp_mesh)
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log
                
                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
                current_lr = optimizers.optimizers[0].param_groups[0]['lr']

                if log_file_handle is not None:
                    metrics = {
                        "step": train_state.step,
                        "mode": "train",
                        "lr": current_lr,
                        "global_avg_loss": global_avg_loss,
                        "mem_max_reserved_gib": gpu_mem_stats.max_reserved_gib,
                    }
                    log_file_handle.write(json.dumps(metrics) + "\n")
                    log_file_handle.flush()

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.red}lr: {current_lr:.6f}  "
                    f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                )

                losses_since_last_log.clear()
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                gpu_memory_monitor.reset_peak_stats()

            checkpoint.save(train_state.step, force=(train_state.step == job_config.training.steps))

            if torch_profiler: torch_profiler.step()
            if memory_profiler: memory_profiler.step()

            if train_state.step == 1:
                utils.set_pg_timeouts(timeout=timedelta(seconds=job_config.comm.train_timeout_seconds), world_mesh=world_mesh)

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)
        if log_file_handle: log_file_handle.close()
        
    logger.info("SimMIM Training completed")

if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()