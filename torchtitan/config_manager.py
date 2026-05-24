# Copyright (c) Emin Orhan.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from collections import defaultdict
from typing import Tuple, Union

import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtitan.logging import logger

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def string_list(raw_arg):
    return raw_arg.split(",")


class JobConfig:
    """
    A helper class to manage the train configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    - if additional explicit cmd args are provided in addition to the toml
    file, they will override the toml config and the argparse defaults

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the section name in the toml file
    followed by name of the option in the toml file. For ex,
    model.size translates to:
        [model]
        size
    in the toml file
    """

    def __init__(self):
        # main parser
        self.parser = argparse.ArgumentParser(description="torchtitan arg parser.")
        self.parser.add_argument("--job.config_file", type=str, default=None, help="Job config file")

        # job level configs
        self.parser.add_argument("--job.dump_folder", type=str, default="./torchtitan/outputs", help="Folder to dump job outputs")
        self.parser.add_argument("--job.description", type=str, default="default job", help="Description of the job")
        self.parser.add_argument("--job.use_for_integration_test", default=False, action="store_true", help="Add this config to the integration test suite")

        # profiling configs
        self.parser.add_argument("--profiling.enable_profiling", action="store_true", help="Whether to enable pytorch profiler")
        self.parser.add_argument("--profiling.save_traces_folder", type=str, default="profile_traces", help="Trace files location")
        self.parser.add_argument("--profiling.profile_freq", type=int, default=10, help="How often to collect profiler traces, in iterations")
        self.parser.add_argument("--profiling.enable_memory_snapshot", action="store_true", default=False, help="Whether to dump memory snapshot")
        self.parser.add_argument("--profiling.save_memory_snapshot_folder", type=str, default="memory_snapshot", help="Memeory snapshot files location")

        # metrics configs
        self.parser.add_argument("--metrics.log_freq", type=int, default=10, help="How often to log training metrics to TensorBoard, in iterations")
        self.parser.add_argument("--metrics.eval_freq", type=int, default=10, help="How often to evaluate model, in iterations")
        self.parser.add_argument("--metrics.enable_color_printing", default=False, action="store_true", help="Whether to enable color printing")
        self.parser.add_argument("--metrics.folder", type=str, default="logs", help="Folder to dump metric logs under")

        # model configs
        self.parser.add_argument("--model.size", type=str, default="2B", help="Model size")
        self.parser.add_argument("--model.img_size", type=int, default=512, help="Size of crops")  # hmmm
        self.parser.add_argument("--model.patch_size", type=int, default=8, help="Patch size")
        self.parser.add_argument("--model.mask_ratio", type=float, default=0.95, help="Mask ratio")
        self.parser.add_argument("--model.backbone", type=str, default="dinov3_vit7b16", help="DINOv3 backbone", choices=["dinov3_vit7b16", "dinov3_vit7b_3D16", "dinov3_vith16plus", "dinov3_vitl16", "dinov3_vitb16", "dinov3_vits16plus", "dinov3_vits16"])
        self.parser.add_argument("--model.head", type=str, default="linear", help="DINOv3 head", choices=["linear", "m2f"])
        self.parser.add_argument("--model.dinov3_repo_folder", type=str, default="/lustre/blizzard/stf218/scratch/emin/dinov3", help="Local path to DINOv3 repo (to be used for model definitions)")
        self.parser.add_argument("--model.num_classes", type=int, default=48, help="Number of classes in output head")
        self.parser.add_argument("--model.crop_size", type=int, nargs='+', default=[512], help="Size of crops (can be [int, int] or [int, int, int])")
        self.parser.add_argument("--model.val_crop_size", type=int, nargs=3, default=[512, 512, 512], help="Size of validation crops ([int, int, int])")
        self.parser.add_argument("--model.use_fa4", default=False, action="store_true", help="Whether to use FlashAttention-4 in model implementation.")

        # data configs
        self.parser.add_argument("--data.dataset_name", type=str, default="cellmap-2d", help="Name of the dataset to load.")
        self.parser.add_argument("--data.dataset_path", type=str, default="eminorhan/cellmap-2d", help="Root directory for the dataset")
        self.parser.add_argument("--data.augment", default=False, action="store_true", help="Whether to use data augmentation (default: False).")

        # optimizer configs
        self.parser.add_argument("--optimizer.name", type=str, default="AdamW", help="Optimizer to use")
        self.parser.add_argument("--optimizer.lr", type=float, default=8e-4, help="Learning rate to use")
        self.parser.add_argument("--optimizer.fused", default=True, action="store_true", help="Whether the fused implementation (CUDA only) is used.")

        # training configs
        self.parser.add_argument("--training.batch_size", type=int, default=8, help="Batch size")
        self.parser.add_argument("--training.num_workers", type=int, default=0, help="Number of data loading workers per DP rank.")
        self.parser.add_argument("--training.warmup_steps", type=int, default=1000, help="Steps for lr scheduler warmup, normally 1/5 of --training.steps")
        self.parser.add_argument("--training.max_norm", type=Union[float, int], default=1.0, help="Max norm for gradient clipping")
        self.parser.add_argument("--training.steps", type=int, default=100000, help="How many train steps to run")
        self.parser.add_argument("--training.data_parallel_replicate_degree", type=int, default=1, help="Degree of data parallelism for weight replication. 1 means disabled. Uses HSDP if both dp_replicate and dp_shard > 1.")
        self.parser.add_argument("--training.data_parallel_shard_degree", type=int, default=-1, help="Degree of data parallelism for weight sharding. 1 means disabled. -1 means leftover ranks will be used. Uses HSDP if both dp_replicate and dp_shard > 1.")
        self.parser.add_argument("--training.tensor_parallel_degree", type=int, default=1, help="Tensor Parallelism degree. 1 means disabled.")
        self.parser.add_argument("--training.enable_loss_parallel", default=True, action="store_true", help="Whether to apply loss parallel when sequence parallel is enabled")
        self.parser.add_argument("--training.mixed_precision_param", type=str, default="bfloat16", choices=["bfloat16", "float32"], help="torch dtype to use for parameters when applying mixed precision via FSDP. This feature only takes effect when data_parallel_degree > 1")
        self.parser.add_argument("--training.mixed_precision_reduce", type=str, default="float32", choices=["float32"], help="torch dtype to use for reductions when applying mixed precision via FSDP. This feature only takes effect when data_parallel_degree > 1")
        self.parser.add_argument("--training.compile", action="store_true", help="Whether to compile the model")
        self.parser.add_argument("--training.gc_freq", type=int, default=50, help="Python garbage control scheduling interval, in steps")
        self.parser.add_argument("--training.seed", type=int, default=None, help="Implement reproducibility by setting a Python, PyTorch and CUDA seed")
        self.parser.add_argument("--training.shuffle_seed", type=int, default=None, help="Random seed to shuffle datasets")

        # experimental configs
        self.parser.add_argument("--experimental.enable_async_tensor_parallel", default=False, action="store_true", help="Whether to apply async tensor parallel (currently only effective when compile is enabled)")
        self.parser.add_argument("--experimental.enable_compiled_autograd", action="store_true", help="Enable CompiledAutograd to compile the backward.")

        # checkpointing configs
        self.parser.add_argument("--checkpoint.enable_checkpoint", action="store_true", help="Whether to enable checkpoint")
        self.parser.add_argument("--checkpoint.folder", type=str, default="checkpoint", help="The folder to store the checkpoints. When enable_checkpoint is set to true, checkpoints will be in {--job.dump_folder}/{--checkpoint.folder}.")
        self.parser.add_argument("--checkpoint.interval_type", type=str, default="steps", help="Checkpointing interval unit of measurement ['step', 'seconds']")
        self.parser.add_argument("--checkpoint.interval", type=int, default=500, help="Checkpointing interval, in steps or seconds depending on --checkpoint.interval_type")
        self.parser.add_argument("--checkpoint.model_weights_only", action="store_true", help="When model_weights_only=True, only model weights will be saved at the end of training. With this, checkpoints can be loaded using `torch.load(..., weights_only=True)` after conversion")
        self.parser.add_argument("--checkpoint.export_dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Converts to the specified precision when training completes and model_weights_only=true. Currently supports float32, float16, and bfloat16. The default value is float32")
        self.parser.add_argument("--checkpoint.create_seed_checkpoint", action="store_true", help="Initializes the full model without applying parallelisms, and then saves it as a seed checkpoint. Note: requires user to call train.py without specifying any parallelisms, e.g. NGPU=1.")
        self.parser.add_argument("--checkpoint.async_mode", type=str, default="disabled", help="Which async checkpoint mode to use: 'disabled': synchronized checkpointing; 'async': torch.distributed.checkpoint.async_save will be used; 'async_with_pinned_mem': this option utilizes a dedicated pinned memory space and creates a separate process for faster GPU->CPU transfer")
        self.parser.add_argument("--checkpoint.keep_latest_k", type=int, default=0, help="Keeps only the latest k checkpoints, and purging older ones. If 0, keep all checkpoints. 0 is the default value.")

        # activation checkpointing configs
        self.parser.add_argument("--activation_checkpoint.mode", type=str, default="selective", help="Type of activation checkpointing to use ['none', 'full', 'selective']")
        self.parser.add_argument("--activation_checkpoint.selective_ac_option", type=str, default="2", help="Selective activation checkpointing options ['int', 'op']. 'int' (e.g., 2) for every nth layer, or 'op' for op level ac.")

        # float8 configs
        self.parser.add_argument("--float8.enable_float8_linear", action="store_true", help="If true, swaps `torch.nn.Linear` with `Float8Linear`. This feature requires you to install 'torchao' which can be found here: https://github.com/pytorch/ao")
        self.parser.add_argument("--float8.enable_fsdp_float8_all_gather", action="store_true", default=False, help="Whether enable float8 all-gather in FSDP")
        self.parser.add_argument("--float8.precompute_float8_dynamic_scale_for_fsdp", action="store_true", default=False, help="Whether precompute float8 scales dynamically for FSDP")
        self.parser.add_argument("--float8.scaling_type_input", type=str, default="dynamic", help="float8 scaling for input, dynamic (default) or delayed", choices=["dynamic", "delayed"])
        self.parser.add_argument("--float8.scaling_type_weight", type=str, default="dynamic", help="float8 scaling for input, dynamic (default) or delayed")
        self.parser.add_argument("--float8.scaling_type_grad_output", type=str, default="dynamic", help="float8 scaling for input, dynamic (default) or delayed")

        # communications library settings
        self.parser.add_argument("--comm.init_timeout_seconds", type=int, default=3600, help="Timeout for communication operations, during initialization and first train step (default: 1 hour).")
        self.parser.add_argument("--comm.train_timeout_seconds", type=int, default=1200, help="Timeout for communication operations after the first train step -- usually a tighter bound than during initialization.")
        self.parser.add_argument("--comm.trace_buf_size", type=int, default=0, help="Flight recorder ring buffer size, >0 means recording by default, 0 means disabled")

        # memory estimation settings
        self.parser.add_argument("--memory_estimation.enabled", help="Whether to estimate memory usage for FSDP", action="store_true")
        self.parser.add_argument("--memory_estimation.disable_fake_mode", help="Whether to estimate memory under FakeTensorMode", default=False, action="store_true")

    def parse_args(self, args_list: list = sys.argv[1:]):
        args, cmd_args = self.parse_args_from_command_line(args_list)
        config_file = getattr(args, "job.config_file", None)
        # build up a two level dict
        args_dict = self._args_to_two_level_dict(args)
        if config_file is not None:
            try:
                with open(config_file, "rb") as f:
                    for k, v in tomllib.load(f).items():
                        # to prevent overwrite of non-specified keys
                        args_dict[k] |= v
            except (FileNotFoundError, tomllib.TOMLDecodeError) as e:
                logger.exception(f"Error while loading the configuration file: {config_file}")
                logger.exception(f"Error details: {str(e)}")
                raise e

        # override args dict with cmd_args
        cmd_args_dict = self._args_to_two_level_dict(cmd_args)
        for section, section_args in cmd_args_dict.items():
            for k, v in section_args.items():
                args_dict[section][k] = v

        for k, v in args_dict.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())
        self._validate_config()

    def _args_to_two_level_dict(self, args: argparse.Namespace) -> defaultdict:
        args_dict = defaultdict(defaultdict)
        for k, v in vars(args).items():
            first_level_key, second_level_key = k.split(".", 1)
            args_dict[first_level_key][second_level_key] = v
        return args_dict

    def _validate_config(self) -> None:
        # TODO: Add more mandatory validations
        assert self.model.size

    def parse_args_from_command_line(self, args_list) -> Tuple[argparse.Namespace, argparse.Namespace]:
        """
        Parse command line arguments and return the parsed args and the command line only args
        """
        args = self.parser.parse_args(args_list)

        # aux parser to parse the command line only args, with no defaults from main parser
        aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        for arg, val in vars(args).items():
            if isinstance(val, bool):
                aux_parser.add_argument("--" + arg, action="store_true" if val else "store_false")
            elif arg == "experimental.pipeline_parallel_split_points":
                # without this special case, type inference breaks here,
                # since the inferred type is just 'list' and it ends up flattening
                # e.g. from ["layers.0", "layers.1"] into ["l", "a", "y", "e", "r", "s", ".0", ...]
                aux_parser.add_argument("--" + arg, type=string_list)
            else:
                aux_parser.add_argument("--" + arg, type=type(val))

        cmd_args, _ = aux_parser.parse_known_args(args_list)

        return args, cmd_args