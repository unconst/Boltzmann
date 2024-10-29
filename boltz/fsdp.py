from collections import defaultdict
from typing import Callable, List, Tuple

import os
import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
import functools
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
import logging

from common import logger

def fsdp_auto_wrap_policy(model, transformer_layer_names):


    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set(transformer_layer_names)
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy


def wrap_model_with_fsdp(
    model: nn.Module,
    device_id: int,
    transformer_layer_names: List[nn.Module],
    is_distributed: bool,
    device: torch.device
) -> nn.Module:
    """
    Wraps a PyTorch model with Fully Sharded Data Parallel (FSDP) if distributed training is enabled.
    
    Args:
        model (nn.Module): The PyTorch model to wrap
        device_id (int): The local device ID for FSDP
        transformer_layer_names (List[nn.Module]): List of transformer layer classes to wrap
        is_distributed (bool): Whether distributed training is enabled
        device (torch.device): The device to move the model to
        
    Returns:
        nn.Module: The wrapped model, either with FSDP or moved to device
        
    Example:
        >>> model = LlamaForCausalLM(config)
        >>> wrapped_model = wrap_model_with_fsdp(
        ...     model=model,
        ...     device_id=local_rank,
        ...     transformer_layer_names=[LlamaDecoderLayer],
        ...     is_distributed=dist.is_initialized(),
        ...     device=torch.device("cuda", local_rank)
        ... )
    """
    if is_distributed:
        # Create the custom auto wrap policy
        auto_wrap_policy = fsdp_auto_wrap_policy(
            model=model,
            transformer_layer_names=transformer_layer_names
        )

        # Wrap with FSDP
        model = FSDP(
            model,
            device_id=device_id, 
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        logger.info(f"Model wrapped with FSDP on device {device} using custom auto wrap policy.")
    else:
        # Move to device for single-process execution
        model = model.to(device)
        logger.info(f"Model moved to device {device}.")
        
    return model

def initialize_distributed_training(
    device_config: str,
) -> Tuple[int, int, int, torch.device]:
    """
    Initializes distributed training configuration and device settings.
    
    Args:
        device_config (str): Device configuration string (e.g. 'cuda:0')
        logger (logging.Logger): Logger instance for output messages
        
    Returns:
        Tuple containing:
            local_rank (int): Local process rank
            global_rank (int): Global process rank  
            world_size (int): Total number of processes
            device (torch.device): Torch device object
            
    Example:
        >>> local_rank, global_rank, world_size, device = initialize_distributed_training(
        ...     device_config='cuda:0',
        ...     logger=logging.getLogger()
        ... )
    """
    # Initialize distributed training if CUDA available and running in distributed mode
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        # Get distributed training parameters from environment
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"]) 
        world_size = int(os.environ["WORLD_SIZE"])

        # Validate GPU availability
        num_gpus = torch.cuda.device_count()
        if local_rank >= num_gpus:
            raise ValueError(f"Local rank {local_rank} exceeds number of available GPUs {num_gpus}.")

        # Configure device
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # Initialize process group
        dist.init_process_group(backend='nccl')
        logger.info(f"Distributed training initialized on rank {global_rank} out of {world_size} processes.")

    else:
        # Single process execution settings
        local_rank = 0
        global_rank = 0 
        world_size = 1
        device = torch.device(device_config)
        logger.warning("Distributed training is not initialized. Running on a single process.")

    return local_rank, global_rank, world_size, device
