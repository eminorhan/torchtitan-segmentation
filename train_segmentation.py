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
from datetime import timedelta

# torch imports
import torch
from torch.distributed.elastic.multiprocessing.errors import record

# torchtitan imports
from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_data_loader
from torchtitan.evaluation import compute_confusion_matrix
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import parallelize_dino, ParallelDims
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.evaluation import evaluate_2d, evaluate_3d

# dino imports
from dinov3.eval.segmentation.models import build_segmentation_decoder


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


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):

    # set up logger
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = utils.Color if job_config.metrics.enable_color_printing else utils.NoColor

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # set determinism, use seed == None to skip deterministic training
    utils.set_determinism(job_config.training.seed)

    # init distributed
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

    # initialize GPU memory monitor and get peak flops for MFU calculation
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(gpu_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    
    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    assert len(job_config.model.crop_size) in (2, 3), f"model.crop_size must have 2 or 3 elements, but got {len(job_config.model.crop_size)}"

    # build dataloaders
    train_loader, val_loader = build_data_loader(
        job_config.training.batch_size,
        job_config.data.dataset_folder,
        tuple(job_config.model.crop_size),
        tuple(job_config.model.val_crop_size),
        dp_rank,
        world_size,
        job_config.data.augment
    )

    # build model skeleton (TODO: maybe try 'meta' init here). NOTE: we load the pretrained weights during ckpt.load() below
    backbone = torch.hub.load(job_config.model.dinov3_repo_folder, job_config.model.backbone, source="local", use_fa4=job_config.model.use_fa4, pretrained=False)
    model = build_segmentation_decoder(backbone, decoder_type=job_config.model.head, num_classes=job_config.model.num_classes)

    if torch.distributed.get_rank() == 0:
        logger.info(f"Model: {model}")

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # parallelization: apply PT-D TP, activation checkpointing, torch.compile, DP
    parallelize_dino(model, world_mesh, parallel_dims, job_config)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"
    model.to(device=init_device)
    model.train()
    model_parts = [model]

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(f"GPU memory usage for model: {gpu_mem_stats.max_reserved_gib:.2f}GiB ({gpu_mem_stats.max_reserved_pct:.2f}%)")
    logger.info(f"Total number of parameters: {utils.get_num_params(model)}")

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

    train_state = TrainState()
    
    # load initial checkpoint
    checkpoint = CheckpointManager(
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint_loaded = checkpoint.load()

    # set up file logger (only on rank 0)
    log_file_handle = None
    if torch.distributed.get_rank() == 0:
        # log file will be under dump_folder/log_folder
        dump_folder = getattr(job_config.job, "dump_folder", ".")
        log_folder = getattr(job_config.metrics, "folder", "logs")
        
        # combine: e.g., "./outputs/dinov3_vitl16_2D_linear_128/logs"
        log_dir = os.path.join(dump_folder, log_folder)
        os.makedirs(log_dir, exist_ok=True)
        
        # define the final file path
        log_file_path = os.path.join(log_dir, "metrics.jsonl")
        
        # open in append mode so resuming jobs simply continue logging
        log_file_handle = open(log_file_path, "a")

    train_iterator = iter(train_loader)
    train_context = get_train_context(parallel_dims.loss_parallel_enabled, job_config.experimental.enable_compiled_autograd)

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    data_loading_times = []
    time_last_log = time.perf_counter()
    gpu_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # cross-entropy loss (same for 2D & 3D)
    def loss_fn(preds, labels):
        return torch.nn.functional.cross_entropy(preds, labels)

    # resampling function
    def resample_preds(preds, labels, crop_size):
        # 2D resampling
        if len(crop_size) == 2:
            if preds.shape[-2:] != labels.shape[-2:]:
                preds = torch.nn.functional.interpolate(input=preds, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        else:  # 3D resampling
            if preds.shape[-3:] != labels.shape[-3:]:
                preds = torch.nn.functional.interpolate(input=preds, size=labels.shape[-3:], mode="trilinear", align_corners=False)
        return preds

    # eval function
    if len(job_config.model.crop_size) == 2:
        eval_fn = evaluate_2d
    else:
        eval_fn = evaluate_3d

    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )

    if torch.distributed.get_rank() == 0:
        utils.print_parameter_status(model)  # check if the parameters are being trained or frozen

    # train loop
    with maybe_enable_profiling(job_config, global_step=train_state.step) as torch_profiler, maybe_enable_memory_snapshot(job_config, global_step=train_state.step) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            inputs, targets = next(train_iterator)
            data_loading_times.append(time.perf_counter() - data_load_start)

            inputs = inputs.cuda()
            targets = targets.cuda()

            optimizers.zero_grad()
            
            # run forward / backward
            with train_context():
                preds = model(inputs)
                # resample predictions if necessary
                preds = resample_preds(preds, targets, job_config.model.crop_size)
                # logger.info(f"train inputs/targets/preds shape: {inputs.shape}/{targets.shape}/{preds.shape}")
                loss = loss_fn(preds, targets)
                # need to free before bwd to avoid peaking memory
                del preds
                loss.backward()
            
            # clip gradients
            for m in model_parts:
                torch.nn.utils.clip_grad_norm_(m.parameters(), job_config.training.max_norm, foreach=True)

            # sync float8 amaxes and scales
            float8_handler.sync_float8_amax_and_scale_history(model_parts)

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            losses_since_last_log.append(loss)

            # ###### log train metrics ######
            if (train_state.step == 1 or train_state.step % job_config.metrics.log_freq == 0):
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = utils.dist_mean(avg_loss, dp_mesh), utils.dist_max(max_loss, dp_mesh)
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log
                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
                current_lr = optimizers.optimizers[0].param_groups[0]['lr']

                # log to file
                if log_file_handle is not None:
                    metrics = {
                        "step": train_state.step,
                        "mode": "train",
                        "lr": current_lr,
                        "global_avg_loss": global_avg_loss,
                        "global_max_loss": global_max_loss,
                        "time_end_to_end_s": time_end_to_end,
                        "time_data_loading_s": time_data_loading,
                        "time_data_loading_pct": time_data_loading_pct,
                        "mem_max_active_gib": gpu_mem_stats.max_active_gib,
                        "mem_max_active_pct": gpu_mem_stats.max_active_pct,
                        "mem_max_reserved_gib": gpu_mem_stats.max_reserved_gib,
                        "mem_max_reserved_pct": gpu_mem_stats.max_reserved_pct,
                        "mem_num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                        "mem_num_ooms": gpu_mem_stats.num_ooms,
                    }
                    log_file_handle.write(json.dumps(metrics) + "\n")
                    log_file_handle.flush()  # force write to disk

                # log to stdout
                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.red}lr: {current_lr:.6f}  "
                    f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                )

                losses_since_last_log.clear()
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                gpu_memory_monitor.reset_peak_stats()

            # ###### eval on val data & visualize results ######
            if train_state.step % job_config.metrics.eval_freq == 0:
                model.eval()
                
                with torch.no_grad():
                    avg_val_loss, avg_miou = eval_fn(model, val_loader, job_config, loss_fn, resample_preds, dp_mesh)
                    
                    # log validation metrics to file
                    if log_file_handle is not None:
                        val_metrics = {
                            "step": train_state.step,
                            "mode": "val",
                            "avg_val_loss": avg_val_loss,
                            "avg_miou": avg_miou
                        }
                        log_file_handle.write(json.dumps(val_metrics) + "\n")
                        log_file_handle.flush()

                    # log to stdout
                    logger.info(
                        f"{color.cyan}step: {train_state.step:2}  "
                        f"{color.green}val loss: {avg_val_loss:.4f}  "
                        f"{color.red}val mIoU: {avg_miou:.4f}  "
                    )

                model.train()
            # ###### end eval & visualize ######

            checkpoint.save(train_state.step, force=(train_state.step == job_config.training.steps))

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(timeout=timedelta(seconds=job_config.comm.train_timeout_seconds), world_mesh=world_mesh)

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    if log_file_handle is not None:
        log_file_handle.close()

    logger.info("Training completed")

if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()