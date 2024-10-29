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
# fmt: off

# Global imports.
import os
import sys 
import time
import wandb
import torch
import random
import asyncio
import argparse
import threading
import traceback
from tqdm import tqdm
import bittensor as bt
from typing import List
import torch.optim as optim
from dotenv import dotenv_values
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import local files.
from boltz.common import *
from boltz.hparams import load_hparams
from boltz.dataset import DatasetLoader

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Validator:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Validator script')
        parser.add_argument('--project', type=str, default='QZWXEC', help='Optional wandb project name')
        parser.add_argument('--netuid', type=int, default=220, help='Bittensor network UID.')
        parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
        parser.add_argument('--actual_batch_size', type=int, default=8, help='Training batch size per accumulation.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., cpu or cuda)')
        parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        config.subtensor.network = 'test'
        config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/'
        if config.debug: debug()
        if config.trace: trace()
        return config

    def __init__(self):
        # Init config.
        self.config = Validator.config()
        logger.info('\n' + '-' * 40 + ' Config ' + '-' * 40)
        logger.info(self.config)

        # Init bittensor objects.
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise ValueError(f'Wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}')
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        logger.info('\n' + '-' * 40 + ' Objects ' + '-' * 40)
        logger.info(f'\nWallet: {self.wallet}\nSubtensor: {self.subtensor}\nMetagraph: {self.metagraph}\nUID: {self.uid}')

        # Init bucket.
        try:
            if self.config.bucket != self.subtensor.get_commitment(self.config.netuid, self.uid):
                raise ValueError('')
        except:
            self.subtensor.commit(self.wallet, self.config.netuid, self.config.bucket)
        logger.info('Bucket:' + self.config.bucket)

        # Init Wandb.
        if self.config.use_wandb:
            # Delete all runs with my name and create a new one.
            try:
                [run.delete() for run in wandb.Api().runs(path=self.config.project)
                 if run.name == f'V{self.uid}' and logger.info(f'Deleting old run: {run}')]
            except: pass
            wandb.init(project=self.config.project, resume='allow', name=f'V{self.uid}', config=self.config)

        # Init model.
        logger.info('\n' + '-' * 40 + ' Hparams ' + '-' * 40)
        self.hparams = load_hparams()
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        #self.model = LlamaForCausalLM(config=self.hparams.model_config)
        self.model = LlamaForCausalLM.from_pretrained('TinyLlama/TinyLlama_v1.1')
        self.model.to(self.config.device)
        self.model.eval()
        
        # Init buckets.
        self.buckets = []
        for uid in self.metagraph.uids:
            # Use --remote to connect to other miners, other wise, only see's config.bucket.
            try: self.buckets.append(self.config.bucket if not self.config.remote else self.subtensor.get_commitment( self.config.netuid, uid ) )
            except: self.buckets.append(None)

        # Init run state.
        self.global_step = 0
        self.last_window = 0
        self.optimal_pages_per_step = 4
        self.current_block = self.subtensor.block
        self.current_window = self.block_to_window( self.current_block )
        self.window_seeds = {self.current_window: self.window_to_seed( self.current_window) }
        self.block_event = asyncio.Event()
        self.new_window_event = asyncio.Event()
        self.stop_event = asyncio.Event()     
        self.loss_change = torch.zeros( 256, dtype = torch.float32 ) 
        self.scores = torch.zeros( 256, dtype = torch.float32 ) 
        self.weights = torch.zeros( 256, dtype = torch.float32 ) 
        self.sample_rate = 1.0
        print ( self.hparams )
        
    async def update(self):
        while not self.stop_event.is_set():                          # Loop until stop_event is set
            self.subtensor = bt.subtensor(config=self.config)        # Reinitialize subtensor with current config
            nxt_meta = self.subtensor.metagraph(self.config.netuid)  # Get the new metagraph for the given netuid
            self.hparams = load_hparams()                            # Reload hyperparameters
            next_buckets = []                                        # Initialize the next_buckets list
            for uid in nxt_meta.uids:                                # Iterate over new metagraph uids
                try: next_buckets.append(self.config.bucket if not self.config.remote else self.subtensor.get_commitment( self.config.netuid, uid ))
                except: next_buckets.append(None)    
            self.buckets = next_buckets                              # Update self.buckets with next_buckets
            for idx, hotkey in enumerate(self.metagraph.hotkeys):    # Iterate over current metagraph hotkeys
                if hotkey != nxt_meta.hotkeys[idx]:                  # Check if hotkey has changed in the new metagraph
                    self.scores[idx] = 0                             # Reset rewards for the changed hotkey
                    self.weights[idx] = 0                            # Reset weights for the changed hotkey
            self.metagraph = nxt_meta                                # Update self.metagraph with new_metagraph
            await asyncio.sleep(60)                                  # Sleep for 60 seconds before the next iteration

    async def run(self):
        # Main loop.
        self.loop = asyncio.get_running_loop()
        self.update_task = asyncio.create_task(self.update())
        self.listener = threading.Thread(target=self.block_listener, args=(self.loop,), daemon=True).start()

        while True:
            
            try:
                # Start step.
                logger.info('\n' + '-' * 40 + f' Step: {self.global_step} ' + '-' * 40)
                step_start_time = time.time()
                self.global_step += 1
                self.step_window = self.current_window
                self.eval_window = self.current_window - 2
                logger.info(f"Step: {self.global_step}, Step Window: {self.step_window}, Eval Window: {self.eval_window}"
                            f"Block: {self.current_block}, Time: {int(step_start_time)}")
                
                # Download the slices for the window.
                logger.info(f"\tDownloading slices from previous window: { self.eval_window }")
                start_time = time.time()
                slice_infos = await download_slices_for_buckets_and_windows(
                    buckets=self.buckets,
                    windows=[self.eval_window]
                )
                await download_slices_for_buckets_and_windows(
                    buckets=self.buckets,
                    windows=[self.eval_window + 1]
                )
                # If there are no slices to eval, wait until the next window then start again.
                if self.eval_window not in slice_infos or len(slice_infos[self.eval_window]) == 0:
                    print ('\t\tNo slices to download, waiting for next window...')
                    while self.current_window == self.step_window: await asyncio.sleep(0.1)
                    continue
                slice_infos = slice_infos[self.eval_window]
                logger.info(f"\t\tDownloaded {len(slice_infos)} slices for previous window: {self.eval_window} in {time.time() - start_time} seconds")
                
                # Step 2: Apply slices to the model from the previous window.
                logger.info(f"\tApplying slices from previous window: {self.eval_window} to model.")
                start_time = time.time()
                eval_slices = await apply_slices_to_model(
                    model=self.model,
                    window=self.eval_window,  # Get files from previous window.
                    seed=self.step_window,  # Use seed as the hash of the current window.
                    compression=self.hparams.compression
                )
                await apply_slices_to_model(
                    model=self.model,
                    window=self.eval_window + 1,  # Get files from previous window.
                    seed=self.step_window,  # Use seed as the hash of the current window.
                    compression=self.hparams.compression
                )
                applied_per_step = len(eval_slices)
                logger.info(f"\t\tApplied {applied_per_step} slices from previous window: {self.eval_window} with seed: {self.window_seeds[self.step_window]} in {time.time() - start_time} seconds")

                indices = await get_indices_for_window(
                    model=self.model,
                    seed=self.window_to_seed(self.eval_window + 1),  # Seed index for the eval window.
                    compression=self.hparams.compression
                )       
                
                # Step 2: Compute slice importance using second-order approximation with Fisher Information Matrix.
                eval_start_time = time.time()         
                info_i = random.choice(slice_infos)
                
                # Get the UID we are evalling.
                try: uid = self.metagraph.hotkeys.index(info_i.hotkey)
                except ValueError:
                    logger.warning(f"Hotkey {info_i.hotkey} not found in metagraph hotkeys.")
                    continue
                
                # Load the slice for the current miner.
                logger.info(f"\tEvalling slice from hotkey: {info_i.hotkey} and uid: {uid}")
                slice_data = await get_slices( info_i.temp_file, self.model.device )
                
                # Load the dataset for this miner.
                start_time = time.time()
                offset_i = self.eval_window * self.hparams.window_length * self.hparams.window_speed
                seed = uid
                sampled_pages = await DatasetLoader.next_pages(
                    offset = offset_i,
                    n_pages = self.hparams.validator_window_eval_size,
                    seed = seed
                )
                random.shuffle(sampled_pages) # Important to not preference early pages.
                logger.info(f"\t\tLoading pages: {[p[1] for p in sampled_pages]} for offset: {offset_i}, uid: {uid} and seed: {seed}")
                eval_dataset = await DatasetLoader.create(
                    batch_size=self.config.actual_batch_size,
                    sequence_length=self.hparams.sequence_length,
                    pages_info = sampled_pages,
                    tokenizer=self.hparams.tokenizer
                )
                logger.info(f"\t\t\tLoaded pages in {time.time() - start_time} seconds")
                
                # Run the eval.
                logger.info(f"\t\tRunning evaluation for uid: {uid} with sample rate: {self.sample_rate}")
                start_time = time.time()
                self.model.zero_grad()
                self.model.eval()
                # Enable gradient computation
                exhuasted_window = False
                full_steps = 0
                with torch.enable_grad():
                    for idx, batch in enumerate(eval_dataset):
                        # Randomly sample every sample_rate examples
                        if random.random() < self.sample_rate:
                            full_steps += 1
                            input_ids = torch.tensor(batch, dtype=torch.long).to(self.model.device)
                            labels = input_ids.clone()
                            labels = torch.where(labels == self.hparams.tokenizer.pad_token_id, -100, labels)
                            with torch.amp.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):  # Enable autocasting
                                outputs = self.model(input_ids=input_ids, labels=labels)
                            loss = outputs.loss
                            loss.backward()
                            if self.step_window != self.current_window:
                                exhuasted_window = True
                                break
                logger.info(f"\t\t\tTotal steps: {idx}, Applied: {full_steps}, Rate: {full_steps/(idx + 1)}, Sample Probability: {self.sample_rate}")
                logger.info(f"\t\t\tFinished running eval with sample rate: {self.sample_rate} on pages in {time.time() - start_time} seconds")
                if exhuasted_window:
                    self.sample_rate = max(0.0001, self.sample_rate * 0.99)
                else:
                    self.sample_rate = min(1, self.sample_rate * 1.01)

                # Collect gradients for all parameters.
                logger.info(f"\t\tComputing scores")
                start_time = time.time()
                gradients = {}
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue
                    gradients[name] = param.grad.view(-1).clone().detach()
                                        
                delta_L = 0.0 
                for name, param in self.model.named_parameters():
                    if param.grad is None:
                        continue  # Skip parameters without gradients
                    # Retrieve the indices for the current parameter subset.
                    param_indices = indices[name].to(self.model.device)
                    # Extract the gradient vector for the current parameter subset.
                    g = gradients[name][param_indices].to(self.model.device)  # Shape: [num_params_in_subset]
                    # Extract and flatten the slice vector for the current parameter subset.
                    s = slice_data[name].view(-1).to(self.model.device)  # Shape: [num_params_in_subset]
                    # Retrieve the current parameter values for the subset.
                    theta = param.data.view(-1)[param_indices]  # Shape: [num_params_in_subset]
                    # Calculate the change in parameter values.
                    delta_theta = theta - s
                    # Compute the cosine similarity between delta_theta and the gradient vector.
                    cosine_similarity = torch.nn.functional.cosine_similarity(delta_theta, gradients[name][param_indices], dim=0).item()
                    # Calculate the weight of the parameter subset.
                    weight = param.data.view(-1)[param_indices].norm().item() + 1e-8
                    # Update the total importance score.
                    delta_L += weight * cosine_similarity

                # Assign the computed importance score to the corresponding UID.
                logger.info(f"\t\t\tAssigning computed importance score to UID: {uid} with score {delta_L}")

                # Clean up GPU memory
                del slice_data
                del eval_dataset
                del gradients
                torch.cuda.empty_cache()
              
                # Step 7: Normalize the scores as rewards and use them as weights.
                start_time = time.time()
                logger.info('\t\t\tWeights:')
                self.loss_change[uid] = delta_L
                self.scores[uid] = (1 - self.hparams.validator_moving_alpha) * delta_L + self.hparams.validator_moving_alpha * self.scores[uid]
                # If a score is NaN, set it to zero
                self.scores[torch.isnan(self.scores)] = 0
                # Get all valid score value indices.
                valid_score_indices = torch.nonzero((self.scores != 0) & (~torch.isnan(self.scores))).squeeze().view(-1, 1)
                # Get all valid score values.
                valid_scores = self.scores[valid_score_indices].view(-1, 1) if valid_score_indices.dim() == 0 else self.scores[valid_score_indices]
                if len(valid_scores) > 0:
                    max_score = torch.max(valid_scores)
                    normalized_scores = torch.softmax((valid_scores - max_score) * self.hparams.validator_weights_temperature, dim=0)
                    self.weights[valid_score_indices] = normalized_scores
                if self.config.use_wandb:
                    for uid_i in valid_score_indices:
                        wandb.log({
                            f"loss_change/{uid_i.item()}": self.loss_change[uid_i].item(),
                            f"moving_scores/{uid_i.item()}": self.scores[uid_i].item(),
                            f"weights/{uid_i.item()}": self.weights[uid_i].item(),
                            'self.sample_rate': self.sample_rate,
                        })
                for uid_i in valid_score_indices:
                    moving_score = self.scores[uid_i].item()
                    weight = self.weights[uid_i].item()
                    step_score = self.loss_change[uid_i].item()
                    logger.info(f"\t\t\t\tuid: {uid_i.item()}, loss_change: {step_score:.6f}, moving_score: {moving_score:.6f}, weight: {weight:.6f}")
                logger.info(f"\t\tFinished evalling uid: {uid} in {time.time() - eval_start_time} seconds")
                
                # Delete lingering files 
                logger.info(f"\tCleaning space.")
                start_time = time.time()
                await delete_files_before_window( window_max = self.current_window - self.hparams.max_history )
                logger.info(f"\t\tFinished cleaning space in {time.time() - start_time} seconds.")
                
                # Ensure window is over.
                logger.info(f'\nGlobal step completed in {time.time() - step_start_time} seconds\n')
                while self.current_window == self.step_window: await asyncio.sleep(0.1)
                                                                
            # Catch keyboard interrrupt.
            except KeyboardInterrupt:
                logger.info("Training interrupted by user. Stopping the run.")
                self.stop_event.set()
                await self.update_task
                sys.exit(0)
            
            # Catch unknown.
            except Exception as e:
                logger.exception(f"Exception during training loop: {e}")
                continue

    # Returns the slice window based on a block.
    def block_to_window(self, block: int) -> int:
        return int(block / self.hparams.window_length)
    
    # Returns the slice window based on a block.
    def window_to_seed(self, window: int) -> int:
        return str( self.subtensor.get_block_hash( window * self.hparams.window_length ) )

    # A listener thread which posts the block event
    # when the chain announces a new block.
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number'])
            loop.call_soon_threadsafe(self.block_event.set)
            if self.block_to_window(self.current_block) != self.current_window:
                self.window_seeds[ self.block_to_window(self.current_block) ] = self.window_to_seed( self.block_to_window(self.current_block) )
                self.current_window = self.block_to_window(self.current_block)
                loop.call_soon_threadsafe(self.new_window_event.set)
                logger.info(f"-- New window: {self.current_window} -- ")
        # Run listener with retry.
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler); break
            except Exception as e:
                 # Wait for 5 seconds before retrying
                logger.error(f"Failed to subscribe to block headers: {e}.\nRetrying in 1 seconds...")
                time.sleep(1) 
            
if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
