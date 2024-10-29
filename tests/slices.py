import os
import socket
import sys
from typing import Dict, List
from filelock import FileLock
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import asyncio
from torch.distributed.fsdp import StateDictType, FullyShardedDataParallel as FSDP
import logging
from torch.distributed import DeviceMesh
import hashlib
import asyncio
import logging
import tempfile
import aiofiles
from dotenv import dotenv_values
import os
import botocore.config
from aiobotocore.session import get_session
import numpy as np
from torch.distributed._shard.sharded_tensor import ShardedTensor

env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = env_config.get("AWS_SECRET_ACCESS_KEY")
# Configure the S3 client
client_config = botocore.config.Config(
    max_pool_connections=256,
)

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        # Create layers as a ModuleDict with numeric keys
        self.layers = nn.ModuleDict(
            {
                str(i): nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                for i in range(num_layers)
            }
        )
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Apply transformer layers sequentially
        for layer_idx in sorted(self.layers.keys(), key=int):  # Sort numerically
            x = self.layers[layer_idx](x)
        return self.fc(x)

async def upload_slice_for_window(
    bucket: str,
    model: nn.Module,
    window: int,
    wallet,
    seed: str,
    compression: int,
    logger: logging.Logger,
) -> str:
    """
    Generates and saves a sliced version of the model's parameters to a local file.

    Args:
        bucket (str): The name of the S3 bucket (not used for local testing).
        model (nn.Module): The FSDP-wrapped model.
        window (int): The current window number.
        wallet: The wallet object containing the hotkey address.
        seed (str): The seed for index generation.
        compression (int): The compression factor.
        logger (logging.Logger): Logger for debug output.

    Returns:
        str: The filename of the saved slice.
    """
    rank = dist.get_rank()
    logger.debug(f"Rank {rank}: Saving slice to local file")

    device = torch.device("cuda", rank)

    # Include the rank in the filename
    filename: str = f"slice-{window}-rank{rank}-{wallet.hotkey.ss58_address}.pt"
    logger.debug(f"Rank {rank}: Filename for slice: {filename}")

    indices: Dict[str, torch.LongTensor] = await get_indices_for_window(
        model=model, seed=seed, compression=compression, logger=logger
    )

    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        model_state_dict = model.state_dict()

    sliced_state_dict: Dict[str, torch.Tensor] = {}
    for name in model_state_dict.keys():
        if name in indices:
            idx = indices[name].to(device)

            # Get local shard of the parameter
            local_shards = model_state_dict[name].local_shards()
            if not local_shards:
                logger.warning(
                    f"Rank {rank}: No local shards for parameter '{name}'. Skipping."
                )
                continue

            local_shard = local_shards[0].tensor.to(device)
            local_shard_flat = local_shard.contiguous().view(-1)
            sliced_param = local_shard_flat[idx]
            sliced_state_dict[name] = sliced_param.cpu()
            logger.debug(
                f"Rank {rank}: Sliced parameter '{name}' with shape {sliced_param.shape}"
            )
        else:
            logger.debug(f"Rank {rank}: Parameter '{name}' not in indices. Skipping.")

    # Save the sliced state dict to a local file
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    torch.save(sliced_state_dict, file_path)
    logger.debug(f"Rank {rank}: Saved sliced state dict to {file_path}")

    return file_path


async def get_indices_for_window(
    model: torch.nn.Module, seed: str, compression: int, logger: logging.Logger
) -> Dict[str, torch.LongTensor]:
    """
    Computes the indices for the given window and compression factor,
    ensuring that the compression is applied correctly across all ranks.

    Args:
        model (torch.nn.Module): The FSDP-wrapped PyTorch model.
        seed (str): The window seed identifier.
        compression (int): The compression factor.
        logger (logging.Logger): Logger for debug statements.

    Returns:
        Dict[str, torch.LongTensor]: A mapping from parameter names to local indices tensors.
    """
    rank = dist.get_rank()
    logger.debug(
        f"Rank {rank}: Computing indices for window seed '{seed}' with compression '{compression}'"
    )

    result = {}

    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        state_dict = model.state_dict()

    for name, param in state_dict.items():
        logger.debug(f"Rank {rank}: Processing parameter: {name}")

        # Check if parameter is a ShardedTensor
        if not isinstance(param, ShardedTensor):
            logger.warning(f"Parameter '{name}' is not a ShardedTensor. Skipping.")
            continue

        # Get total size of the parameter from metadata
        param_metadata = param.metadata()  # Call the metadata function
        global_size = param_metadata.size  # Global size of the parameter

        # Compute the total number of elements
        total_param_size = 1
        for dim_size in global_size:
            total_param_size *= dim_size

        if total_param_size <= 0:
            logger.warning(f"Parameter '{name}' has no elements. Skipping.")
            continue

        # Compute the total number of indices to select
        num_indices = max(1, total_param_size // compression)
        num_indices = min(num_indices, total_param_size)

        # Generate the same global indices on all ranks
        seed_str = f"{seed}_{name}"
        seed_int = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed_int)

        global_indices = rng.choice(total_param_size, size=num_indices, replace=False)
        global_indices.sort()

        # Get local shard metadata
        local_shards = param.local_shards()
        if not local_shards:
            logger.warning(
                f"Rank {rank}: No local shards for parameter '{name}'. Skipping."
            )
            continue

        local_shard_metadata = local_shards[0].metadata
        shard_offsets = local_shard_metadata.shard_offsets
        shard_sizes = local_shard_metadata.shard_sizes

        # Assuming the parameter is flattened (1D), adjust the indices accordingly
        shard_start_idx = shard_offsets[0]
        shard_end_idx = shard_start_idx + shard_sizes[0]

        # Find indices that are within the local shard
        mask = (global_indices >= shard_start_idx) & (global_indices < shard_end_idx)
        local_indices = global_indices[mask] - shard_start_idx

        if local_indices.size == 0:
            logger.debug(
                f"Rank {rank}: No indices for parameter '{name}' in local shard."
            )
            continue

        indices_tensor = torch.from_numpy(local_indices).long()
        result[name] = indices_tensor

        logger.debug(
            f"Rank {rank}: Generated {len(indices_tensor)} local indices for parameter '{name}'"
        )

    if not result:
        logger.warning(
            f"Rank {rank}: No indices generated. Slice will not be uploaded."
        )
        return {}

    return result


async def apply_slices_to_model(
    model: torch.nn.Module,
    window: int,
    seed: str,
    compression: int,
    logger: logging.Logger,
    key: str = "slice",
) -> List[str]:
    """
    Applies slices from a specific window to the given FSDP model.

    Args:
        model (torch.nn.Module): The FSDP-wrapped model to which the slices will be applied.
        window (int): The window identifier.
        seed (str): The seed used for generating indices.
        compression (int): The compression factor.
        logger (logging.Logger): Logger for debug output.
        key (str): Key prefix for slice files.

    Returns:
        List[str]: A list of all the slice files that were applied.
    """
    rank = dist.get_rank()
    logger.debug(
        f"Rank {rank}: Applying slices for window {window} with seed '{seed}' and compression '{compression}'"
    )

    # Get indices for this rank's parameters
    indices_dict = await get_indices_for_window(model, seed, compression, logger)

    # Load slices specific to this rank
    slice_files = await load_files_for_window_and_rank(
        window=window, rank=rank, key=key, logger=logger
    )

    if not slice_files:
        logger.warning(
            f"Rank {rank}: No slice files found for window {window} and rank {rank}"
        )
        return []

    device = torch.device("cuda", rank)
    logger.debug(f"Rank {rank}: Using device {device}")

    # Dictionaries to accumulate the sum of values and count per parameter
    param_sums: Dict[str, torch.Tensor] = {}
    slices_per_param: Dict[str, int] = {}

    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        model_state_dict = model.state_dict()

    for name in model_state_dict.keys():
        # Get local shard of the parameter
        local_shards = model_state_dict[name].local_shards()
        if not local_shards:
            logger.warning(
                f"Rank {rank}: No local shards for parameter '{name}'. Skipping."
            )
            continue

        local_shard = local_shards[0].tensor.to(device)
        param_sums[name] = torch.zeros_like(local_shard)
        slices_per_param[name] = 0

    # Iterate over each slice file and compute the sum of values
    for file_i in slice_files:
        try:
            slice_i = torch.load(file_i, map_location=device)
            for name in param_sums.keys():
                if name not in indices_dict or name not in slice_i:
                    continue
                values = slice_i[name].to(device)
                param_indices = indices_dict[name].to(device)
                param_sums[name].view(-1)[param_indices] += values
                slices_per_param[name] += 1
            del slice_i
        except Exception as e:
            logger.exception(f"Rank {rank}: Error applying slice from {file_i}: {e}")

    # Apply the average to the parameters
    for name in param_sums.keys():
        if slices_per_param[name] == 0:
            continue
        param_indices = indices_dict[name].to(device)
        avg_param = param_sums[name].view(-1)[param_indices] / slices_per_param[name]

        # Update the local shard of the parameter within no_grad
        with torch.no_grad():
            local_shards = model_state_dict[name].local_shards()
            local_shard = local_shards[0].tensor.to(device)
            local_shard_flat = local_shard.view(-1)
            local_shard_flat.index_copy_(
                0, param_indices, avg_param.to(local_shard_flat.dtype)
            )

    # Load the updated state dict back into the model
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        model.load_state_dict(model_state_dict, strict=False)
        logger.debug(f"Rank {rank}: Loaded updated state dict into model.")

    return slice_files


async def load_files_for_window_and_rank(
    window: int, rank: int, logger, key: str = "slice"
) -> List[str]:
    """
    Retrieves the paths to downloaded window files for a specific rank from the temporary directory.

    Args:
        window (int): The window identifier.
        rank (int): The rank identifier.

    Returns:
        List[str]: A list of file paths corresponding to the window and rank.
    """
    logger.debug(
        f"Retrieving files for window {window} and rank {rank} from temporary directory"
    )
    temp_dir = tempfile.gettempdir()
    window_files = []
    file_pattern = f"{key}-{window}-rank{rank}-"
    for filename in os.listdir(temp_dir):
        if filename.startswith(file_pattern) and filename.endswith(".pt"):
            file_path = os.path.join(temp_dir, filename)
            window_files.append(file_path)
            logger.debug(f"Found file {filename} for window {window} and rank {rank}")
    return window_files
def setup(rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    logger = logging.getLogger(f"Rank {rank}")
    logger.debug(f"Process group initialized. Using GPU {rank}")


def setup_logging(rank: int) -> logging.Logger:
    """
    Sets up a logger for the given rank.

    Args:
        rank (int): The rank of the current process.

    Returns:
        logging.Logger: Configured logger for the rank.
    """
    logger = logging.getLogger(f"Rank {rank}")
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create File Handler
    file_handler = logging.FileHandler(f"log_rank_{rank}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create Stream Handler (optional, can be removed if not needed)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    # Avoid adding multiple handlers to the logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def cleanup(rank):
    logger = logging.getLogger(f"Rank {rank}")
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.debug("Process group destroyed.")
    else:
        logger.warning("Process group not initialized; skipping cleanup.")

def run_fsdp(rank, world_size, master_port):
    logger = logging.getLogger(f"Rank {rank}")
    try:
        logger = setup_logging(rank)
        logger.info(f"Running on rank {rank}")
        setup(rank, world_size, master_port)

        model = SimpleTransformer().to(rank)
        # Count and log the number of parameters in the model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"Total parameters in the model: {total_params:,}")
        logger.info(f"Trainable parameters in the model: {trainable_params:,}")

        fsdp_model = FSDP(model)
        seed = "test_seed"
        compression = 10000
        window = 125

        class MockWallet:
            class hotkey:
                ss58_address = f"test_address_{rank}"

        mock_wallet = MockWallet()

        # Generate and save slices
        filename = asyncio.run(
            upload_slice_for_window(
                bucket="decis",  # Not used in this context
                model=fsdp_model,
                window=window,
                wallet=mock_wallet,
                seed=seed,
                compression=compression,
                logger=logger,
            )
        )

        logger.debug(f"Rank {rank}: Slice saved to {filename}")

        # Apply slices to the model
        slice_files = asyncio.run(
            apply_slices_to_model(
                model=fsdp_model,
                window=window,
                seed=seed,
                compression=compression,
                logger=logger,
            )
        )

        logger.debug(f"Rank {rank}: Applied slices: {slice_files}")

    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
    finally:
        cleanup(rank)


def get_available_gpus():
    return torch.cuda.device_count()


def find_free_port():
    """Finds a free port on the localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    world_size = 4
    available_gpus = get_available_gpus()
    if available_gpus < world_size:
        print(f"Insufficient GPUs. Available: {available_gpus}, Required: {world_size}")
        sys.exit(1)

    # Dynamically find a free port to avoid EADDRINUSE
    master_port = find_free_port()
    print(f"Starting FSDP with world_size={world_size} on port {master_port}")

    mp.spawn(run_fsdp, args=(world_size, master_port), nprocs=world_size, join=True)
