# The MIT License (MIT)
# © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import io
import sys 
import uuid
import time
import fcntl
import torch
import uvloop
import hashlib
import asyncio
import logging
import tempfile
import aiofiles
import numpy as np
import aiobotocore
import bittensor as bt
import botocore.config
from typing import List, Dict
from dotenv import dotenv_values
from types import SimpleNamespace
from rich.logging import RichHandler
from filelock import FileLock, Timeout
from aiobotocore.session import get_session
from rich.highlighter import NullHighlighter
import torch
import hashlib
import numpy as np
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
import torch.distributed as dist
from torch import nn
import traceback

# Configure loguru logger
FORMAT = "%(message)s"
logging.basicConfig( 
    level=logging.INFO, 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[
        RichHandler(
            markup=True, 
            rich_tracebacks=True, 
            highlighter=NullHighlighter(),
            show_level=False,
            show_time=False,
            show_path=False
        )
    ]
)
logger = logging.getLogger("rich")
logger.setLevel(logging.INFO)
def debug():
    logger.setLevel(logging.DEBUG)
def trace():
    logger.setLevel(logging.TRACE)
# Log helper.
def T(): return time.time()
def P( w, d ): return f"[steel_blue]{w}[/steel_blue] ([grey63]{d:.2f}s[/grey63])"

# Load environment variables
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')

# Configure the S3 client
client_config = botocore.config.Config(
    max_pool_connections=256,
)

# Set uvloop as the event loop policy
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Define a semaphore to limit concurrent downloads (adjust as needed)
semaphore = asyncio.Semaphore(1000)

async def get_slices( filename:str, device:str ) -> Dict[str, torch.Tensor]:
    # Attempt to acquire the lock with a timeout of 1 second.
    lock: FileLock = FileLock(f"{filename}.lock")
    with lock.acquire(timeout=5):
        pass
    return torch.load(
        filename,
        map_location=torch.device(device),
        weights_only = True,
    )

async def apply_slices_to_model(
    model: nn.Module,
    window: int,
    seed: str,
    compression: int,
    key: str = 'slice'
) -> List[str]:
    """
    Applies slices from a specific window to the given FSDP model.

    Args:
        model (torch.nn.Module): The FSDP-wrapped PyTorch model to which the slices will be applied.
        window (int): The window identifier.
        seed (str): The seed used for generating indices.
        compression (int): The compression factor.
        key (str): The key used to identify the slices.

    Returns:
        List[str]: A list of all the slice files that were applied.

    Example:
        slice_files = await apply_slices_to_model(
            model=my_fsdp_model,
            window=42,
            seed="1234",
            compression=10,
            key='slice',
        )

    Notes:
        - This function is adapted to work with FSDP. It ensures that all ranks participate
          in collective operations required by FSDP to prevent hangs.
        - Exception handling is added to ensure that any errors are caught, and all ranks exit gracefully.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.debug(f"Rank {rank}: Starting apply_slices_to_model")

    # Get indices associated with the window (all ranks must participate)
    try:
        indices_dict: Dict[str, torch.LongTensor] = await get_indices_for_window(model, seed, compression)
    except Exception as e:
        logger.exception(f"Rank {rank}: Failed to get indices: {e}")
        sys.exit(1)  # Ensure all ranks exit to prevent hangs

    logger.debug(f"Rank {rank}: Retrieved indices for parameters")

    # Load slice files on rank 0 and broadcast the list to all ranks
    if rank == 0:
        try:
            slice_files: List[str] = await load_files_for_window(window=window, key=key)
            logger.debug(f"Rank {rank}: Loaded {len(slice_files)} slice files")
        except Exception as e:
            logger.exception(f"Rank {rank}: Failed to load slice files: {e}")
            slice_files = []
    else:
        slice_files = []

    # # Broadcast the slice_files list to all ranks
    # slice_files_list = [slice_files]
    # dist.broadcast_object_list(slice_files_list, src=0)
    # slice_files = slice_files_list[0]

    if not slice_files:
        logger.warning(f"Rank {rank}: No slice files to process for window {window}")
        return slice_files  # Early return, but all ranks have synchronized here

    # Initialize dictionaries to keep track of sums and counts
    param_sums: Dict[str, torch.Tensor] = {}
    slices_per_param: Dict[str, int] = {}

    # Rank 0 processes the slice files and reconstructs the parameters
    if rank == 0:
        for name in indices_dict.keys():
            param_sums[name] = torch.zeros(indices_dict[name].numel(), dtype=torch.float32)
            slices_per_param[name] = 0

        # Process each slice file
        for file_i in slice_files:
            logger.debug(f"Rank {rank}: Processing slice file {file_i}")
            try:
                slice_i = await get_slices(file_i, 'cpu')  # Load slices to CPU
                for name in slice_i.keys():
                    if name in param_sums:
                        param_sums[name] += slice_i[name].cpu()
                        slices_per_param[name] += 1
            except Exception as e:
                logger.exception(f"Rank {rank}: Error processing slice file {file_i}: {e}")
                continue

        # Average the sums to get the updated parameters
        for name in param_sums.keys():
            if slices_per_param[name] > 0:
                param_sums[name] /= slices_per_param[name]
            else:
                logger.warning(f"Rank {rank}: No slices applied for parameter {name}")

    # Broadcast the param_sums and slices_per_param to all ranks
    # param_sums_list = [param_sums]
    # slices_per_param_list = [slices_per_param]
    # dist.broadcast_object_list(param_sums_list, src=0)
    # dist.broadcast_object_list(slices_per_param_list, src=0)
    # param_sums = param_sums_list[0]
    # slices_per_param = slices_per_param_list[0]

    # All ranks participate in updating the model parameters
    try:
        # Retrieve the full state_dict (all ranks must participate)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict: Dict[str, torch.Tensor] = model.state_dict()

        for name, param in state_dict.items():
            if name in indices_dict and name in param_sums and slices_per_param[name] > 0:
                indices = indices_dict[name].to('cpu')
                updated_values = param_sums[name].to(param.dtype)
                # Update the parameter values at the specified indices
                param.view(-1)[indices] = updated_values
                logger.trace(f"Rank {rank}: Updated parameter {name}")
            else:
                logger.trace(f"Rank {rank}: No updates applied to parameter {name}")

        # Broadcast the updated state_dict from rank 0 to all other ranks
        state_dict_list = [state_dict]
        dist.broadcast_object_list(state_dict_list, src=0)
        state_dict = state_dict_list[0]
        logger.debug(f"Rank {rank}: Received updated state_dict from broadcast")

        # Load the updated state_dict back into the model (all ranks must participate)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state_dict, strict=False)
        logger.debug(f"Rank {rank}: Model parameters updated")

    except Exception as e:
        logger.exception(f"Rank {rank}: Failed to update model parameters: {e}")
        sys.exit(1)  # Ensure all ranks exit

    return slice_files

async def upload_slice_for_window(
    bucket: str,
    model: torch.nn.Module,
    window: int,
    seed: str,
    wallet: 'bt.wallet',
    compression: int,
    key: str = 'slice'
):
    """
    Uploads a compressed slice of an FSDP model to a storage bucket.

    Args:
        bucket (str): Name of the storage bucket.
        model (torch.nn.Module): The FSDP-wrapped PyTorch model to be sliced and uploaded.
        window (int): The window identifier.
        seed (str): The seed used for generating indices.
        wallet (bt.wallet): The wallet object containing the hotkey.
        compression (int): The compression factor.
        key (str): The key used to identify the slices.

    Returns:
        None

    Example Usage:
        await upload_slice_for_window(
            bucket='my-bucket',
            model=my_fsdp_model,
            window=42,
            seed='1234',
            wallet=my_wallet,
            compression=10,
            key='slice'
        )

    Notes:
        - This function ensures that all ranks participate in necessary collective operations with FSDP.
        - Only Rank 0 performs the actual upload to the storage bucket.
        - All ranks must participate in collective operations to prevent hangs.
    """
    rank = dist.get_rank()
    logger.debug(f"Rank {rank}: Starting upload_slice_for_window")

    # Generate the filename based on the window and wallet hotkey
    filename = f'{key}-{window}-{wallet.hotkey.ss58_address}.pt'
    logger.debug(f"Rank {rank}: Filename for slice: {filename}")

    try:
        # Get indices for slicing (all ranks must participate)
        indices = await get_indices_for_window(model, seed, compression)
        logger.debug(f"Rank {rank}: Retrieved indices for slicing")

        # Retrieve the full state_dict (all ranks must participate)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
        logger.debug(f"Rank {rank}: Retrieved state_dict for slicing")

        # Prepare the sliced state_dict
        slice_dict = {}
        for name, param in state_dict.items():
            if name in indices:
                param_indices = indices[name].to('cpu')
                sliced_param = param.view(-1)[param_indices].cpu()
                slice_dict[name] = sliced_param
                logger.trace(f"Rank {rank}: Sliced parameter {name}")
            else:
                logger.trace(f"Rank {rank}: No indices for parameter {name}; skipping")
    except Exception as e:
        logger.exception(f"Rank {rank}: Error during slicing: {e}")
        sys.exit(1)  # Ensure all ranks exit to prevent hangs

    # Only Rank 0 saves and uploads the slice
    if rank == 0:
        try:
            # Save the slice_dict to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                torch.save(slice_dict, temp_file)
                temp_file_name = temp_file.name
            logger.debug(f"Rank {rank}: Saved slice to temporary file {temp_file_name}")

            # Initialize S3 client
            session = get_session()
            async with session.create_client(
                's3',
                region_name='us-east-1',  # Replace with your region
                config=client_config,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            ) as s3_client:
                try:
                    # Upload the file to S3
                    with open(temp_file_name, 'rb') as f:
                        await s3_client.put_object(Bucket=bucket, Key=filename, Body=f)
                    logger.debug(f"Rank {rank}: Uploaded slice to bucket {bucket} with key {filename}")

                    # Optionally set object ACL to public-read
                    await s3_client.put_object_acl(Bucket=bucket, Key=filename, ACL='public-read')
                    logger.debug(f"Rank {rank}: Set object ACL to public-read for {filename}")
                except Exception as e:
                    logger.exception(f"Rank {rank}: Failed to upload slice to storage: {e}")
                finally:
                    # Clean up the temporary file
                    os.remove(temp_file_name)
                    logger.debug(f"Rank {rank}: Removed temporary file {temp_file_name}")
        except Exception as e:
            logger.exception(f"Rank {rank}: Error during saving or uploading slice: {e}")
            sys.exit(1)  # Ensure all ranks exit to prevent hangs
    else:
        logger.debug(f"Rank {rank}: Slice preparation complete. Waiting for Rank 0 to upload.")

    # Synchronize all ranks to ensure upload is completed before proceeding
    dist.barrier()
    logger.debug(f"Rank {rank}: Completed upload_slice_for_window")

async def get_indices_for_window(model: torch.nn.Module, seed: str, compression: int) -> Dict[str, torch.LongTensor]:
    """
    Computes the indices for the given window and compression factor.

    Args:
        model (torch.nn.Module): The PyTorch model.
        seed (str): The window seed identifier.
        compression (int): The compression factor.

    Returns:
        Dict[str, torch.LongTensor]: A dictionary mapping parameter names to index tensors.
    """
    logger.debug(f"Starting get_indices_for_window with seed={seed}, compression={compression}")
    result = {}

    # Seed the random number generator with the seed
    seed_int = int(hashlib.md5(str(seed).encode('utf-8')).hexdigest(), 16) % (2**32)
    logger.trace(f"Converted seed '{seed}' to integer: {seed_int}")
    rng = np.random.default_rng(seed_int)
    logger.trace(f"Initialized random number generator with seed: {seed_int}")

    # Retrieve the full state dict to get parameter shapes
    logger.trace("Retrieving full state dict")
    dist.barrier()
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
    logger.trace(f"Retrieved state dict with {len(state_dict)} parameters")

    # For each parameter, compute the indices
    for name, param in state_dict.items():
        logger.trace(f"Processing parameter: {name}")
        numel = param.numel()
        logger.trace(f"Parameter {name} has {numel} elements")
        num_indices = max(1, int(numel // compression))
        logger.trace(f"Selecting {num_indices} indices for parameter {name}")
        indices = rng.choice(numel, size=num_indices, replace=False)
        logger.trace(f"Generated indices for {name}: min={indices.min()}, max={indices.max()}, shape={indices.shape}")
        result[name] = torch.from_numpy(indices).long().cpu()
        logger.trace(f"Converted indices for {name} to torch.LongTensor on CPU")

    logger.trace(f"Finished get_indices_for_window, returning dict with {len(result)} entries")
    return result


async def download_file(s3_client, bucket: str, filename: str) -> str:
    """
    Downloads a file from S3, using parallel downloads for large files.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        filename (str): The S3 object key (filename).

    Returns:
        str: The path to the downloaded file in the temporary directory.
    """
    async with semaphore:
        temp_file = os.path.join(tempfile.gettempdir(), filename)
        # Check if the file exists.
        if os.path.exists(temp_file):
            logger.debug(f"File {temp_file} already exists, skipping download.")
            return temp_file
        lock_file = f"{temp_file}.lock"
        lock = FileLock(lock_file)
        try:
            # Try to acquire both locks with a timeout
            with lock.acquire(timeout=1):
                # Proceed to download the file
                logger.debug(f"Downloading file {filename} to {temp_file}")
                head_response = await s3_client.head_object(Bucket=bucket, Key=filename)
                object_size = head_response['ContentLength']
                CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB

                response = await s3_client.get_object(Bucket=bucket, Key=filename)
                async with aiofiles.open(temp_file, 'wb') as outfile:
                    while True:
                        chunk = await response['Body'].read(CHUNK_SIZE)
                        if not chunk:
                            break
                        await outfile.write(chunk)

                logger.debug(f"Successfully downloaded file {filename} to {temp_file}")
                return temp_file

        except Timeout:
            logger.error(f"Timeout occurred while trying to acquire lock on {lock_file}")
            return None
        except Exception as e:
            logger.exception(f"Failed to download file {filename} from bucket {bucket}: {e}")
            return None
        finally:
            # The lock is automatically released when exiting the 'with' block
            pass

async def handle_file(s3_client, bucket: str, filename: str, hotkey: str, window: int):
    """
    Handles downloading a single file from S3.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        filename (str): The S3 object key (filename).
        hotkey (str): The hotkey identifier.
        window (int): The window identifier.

    Returns:
        SimpleNamespace: An object containing file metadata and the path to the downloaded file.
    """
    logger.debug(f"Handling file {filename} for window {window} and hotkey {hotkey}")
    temp_file = await download_file(s3_client, bucket, filename)
    if temp_file:
        return SimpleNamespace(bucket=bucket, hotkey=hotkey, filename=filename, window=window, temp_file=temp_file)
    return None

async def process_bucket(s3_client, bucket: str, windows: List[int], key:str = 'slice'):
    """
    Processes an S3 bucket to download files matching the given windows.

    Args:
        s3_client: The S3 client.
        bucket (str): Name of the S3 bucket.
        windows (List[int]): A list of window identifiers.

    Returns:
        List[SimpleNamespace]: A list of file metadata and paths for downloaded files.
    """
    logger.debug(f"Processing bucket {bucket} for window {windows}")
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')

    for window in windows:
        prefix = f'{key}-{window}'
        logger.debug(f"Listing objects with prefix {prefix}")
        async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            logger.trace(f"Processing page for prefix {prefix}")
            if 'Contents' not in page:
                logger.trace(f"No contents found for prefix {prefix}")
                continue
            download_tasks = []
            for obj in page.get('Contents', []):
                filename = obj['Key']
                logger.trace(f"Processing object with key {filename}")
                try:
                    parts = filename.split('-')
                    slice_window = int(parts[1])
                    slice_hotkey = parts[2].split('.')[0]
                    logger.trace(f"Parsed filename {filename} into window {slice_window} and hotkey {slice_hotkey}")
                    if slice_window == window:
                        download_tasks.append(handle_file(s3_client, bucket, filename, slice_hotkey, slice_window))
                except Exception:
                    logger.exception(f"Error processing filename {filename}")
                    continue
            # Download the files concurrently
            results = await asyncio.gather(*download_tasks)
            files.extend([res for res in results if res])
            logger.trace(f"Completed processing page for prefix {prefix}")
    logger.trace(f"Completed processing bucket {bucket} for windows {windows}")
    return files

async def download_slices_for_buckets_and_windows(buckets: List[str], windows: List[int], key:str = 'slice') -> Dict[int, List[SimpleNamespace]]:
    """
    Downloads files from multiple S3 buckets for the given windows.

    Args:
        buckets (List[str]): A list of S3 bucket names.
        windows (List[int]): A list of window identifiers.

    Returns:
        Dict[int, List[SimpleNamespace]]: A dictionary mapping windows to lists of file metadata and paths.
    """
    logger.debug(f"Downloading files for buckets {set(buckets)} and windows {windows}")
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        tasks = []
        for bucket in set(buckets):
            if not bucket:
                continue
            tasks.append(process_bucket(s3_client, bucket, windows, key))
        results = await asyncio.gather(*tasks)
        # Flatten the list of lists
        files = [item for sublist in results for item in sublist]

        # Create a dictionary with windows as keys and list of files as values
        windows_dict = {}
        for file in files:
            window = file.window
            if window not in windows_dict:
                windows_dict[window] = []
            windows_dict[window].append(file)

        logger.debug(f"Downloaded all files grouped by windows: {windows}")
        return windows_dict

async def load_files_for_window(window: int, key: str = 'slice') -> List[str]:
    """
    Retrieves the paths to downloaded window files from the temporary directory.

    Args:
        window (int): The window identifier.

    Returns:
        List[str]: A list of file paths corresponding to the window.
    """
    logger.debug(f"Retrieving files for window {window} from temporary directory")
    temp_dir = tempfile.gettempdir()
    window_files = []
    for filename in os.listdir(temp_dir):
        if filename.startswith(f"{key}-{window}-") and filename.endswith(".pt"):
            window_files.append(os.path.join(temp_dir, filename))
            logger.debug(f"Found file {filename} for window {window}")
    return window_files

async def delete_files_before_window(window_max: int, key:str = 'slice'):
    """
    Deletes all files on the local machine which have a window id before a specific value window_max.

    Args:
        window_max (int): The maximum window id. Files with window ids less than this value will be deleted.
    """
    logger.debug(f"Deleting files with window id before {window_max}")
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith(f"{key}-") and ( filename.endswith(".pt") or filename.endswith(".lock") ):
            try:
                parts = filename.split('-')
                window_id = int(parts[1])
                if window_id < window_max:
                    if os.path.exists(filename):
                        os.remove(filename)
                        logger.debug(f"Deleted file {filename}")
            except Exception as e:
                logger.error(f"Error deleting file {filename}: {e}")

async def delete_files_from_bucket_before_window(bucket: str, window_max: int, key: str = 'slice'):
    """
    Deletes all files in the specified S3 bucket which have a window id before a specific value window_max.

    Args:
        bucket (str): The name of the S3 bucket.
        window_max (int): The maximum window id. Files with window ids less than this value will be deleted.
    """
    logger.debug(f"Deleting files in bucket {bucket} with window id before {window_max}")
    session = get_session()
    async with session.create_client(
        's3',
        region_name='us-east-1',
        config=client_config,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    ) as s3_client:
        try:
            response = await s3_client.list_objects_v2(Bucket=bucket)
            if 'Contents' in response:
                for obj in response['Contents']:
                    filename = obj['Key']
                    if filename.startswith(f"{key}-") and filename.endswith(".pt"):
                        try:
                            parts = filename.split('-')
                            window_id = int(parts[1])
                            if window_id < window_max:
                                await s3_client.delete_object(Bucket=bucket, Key=filename)
                                logger.debug(f"Deleted file {filename} from bucket {bucket}")
                        except Exception as e:
                            logger.error(f"Error deleting file {filename} from bucket {bucket}: {e}")
        except Exception as e:
            logger.error(f"Error listing objects in bucket {bucket}: {e}")
