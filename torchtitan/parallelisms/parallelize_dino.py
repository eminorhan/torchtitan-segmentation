# Copyright (c) Emin Orhan.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard, distribute_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging import logger
from torchtitan.parallelisms.parallel_dims import ParallelDims
from torch.distributed.tensor import DTensor


def parallelize_dino(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise, the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        if job_config.experimental.enable_async_tensor_parallel and not job_config.training.compile:
            raise RuntimeError("Async TP requires --training.compile")
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8=job_config.float8.enable_float8_linear,
            enable_async_tp=job_config.experimental.enable_async_tensor_parallel,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if job_config.training.compile:
        apply_compile(model)

    if parallel_dims.dp_enabled:
        if parallel_dims.dp_shard_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mesh = world_mesh["dp_replicate", "dp_shard"]
            else:
                dp_mesh = world_mesh["dp"]

            apply_fsdp(
                model,
                dp_mesh,
                param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
                reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            )
            
            if parallel_dims.dp_replicate_enabled:
                logger.info("Applied HSDP to the model")
            else:
                logger.info("Applied FSDP to the model")
        else:
            if world_mesh.ndim > 1:
                raise RuntimeError("DDP has not supported > 1D parallelism")
            
            apply_ddp(
                model,
                world_mesh,
                enable_compile=job_config.training.compile,
                enable_compiled_autograd=job_config.experimental.enable_compiled_autograd,
            )


def apply_tp(
    model: nn.Module,
    tp_mesh: "DeviceMesh",
    loss_parallel: bool,
    enable_float8: bool,
    enable_async_tp: bool,
):
    """
    Apply Tensor Parallelism to the MLP blocks of the FeatureDecoder ViT model.
    """
    feature_model = model.segmentation_model[0].feature_model

    # 1. Parallelize individual SelfAttentionBlocks
    for block in feature_model.blocks:
        layer_tp_plan = {
            "mlp.fc1": ColwiseParallel(),
            "mlp.fc2": RowwiseParallel(),
        }
        parallelize_module(block, tp_mesh, layer_tp_plan)

    logger.info("Applied Tensor Parallelism to the model")

# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def _apply_ac_to_transformer_block(module: nn.Module, ac_config):
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}")

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(f"Invalid selective AC option: {ac_config.selective_ac_option}. Valid options: 'op' or a positive int representing layer frequency")
    if use_op_sac:
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0)
                return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(ac_config.selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.segmentation_model[0].feature_model.blocks.named_children():  # apply AC to SelfAttentionBlocks TODO: check this
        transformer_block = _apply_ac_to_transformer_block(transformer_block, ac_config)
        model.segmentation_model[0].feature_model.blocks.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each SelfAttentionBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    for layer_id, transformer_block in model.segmentation_model[0].feature_model.blocks.named_children():  # apply torch.compile to SelfAttentionBlocks TODO: check this
        transformer_block = torch.compile(transformer_block, mode="default", fullgraph=True)
        model.segmentation_model[0].feature_model.blocks.register_module(layer_id, transformer_block)

    logger.info("Compiling each SelfAttentionBlock with torch.compile")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

    fsdp_config = {
        "mesh": dp_mesh, 
        "mp_policy": mp_policy
    }

    for transformer_block in model.segmentation_model[0].feature_model.blocks:  # TODO: is this correct?
        fully_shard(transformer_block, **fsdp_config, reshard_after_forward=False)
        
    fully_shard(model, **fsdp_config, reshard_after_forward=False)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=2048)

    logger.info("Applied DDP to the model")