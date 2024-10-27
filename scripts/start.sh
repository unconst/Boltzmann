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

# Close down all previous processes and restart them.
pm2 sendSignal SIGINT all
pm2 delete all
# Delete items from bucket
BUCKET=${1:-decis}
PROJECT=${2:-aesop}
python3 tools/clean.py --bucket $BUCKET

# Start all the processes again.
pm2 start validator.py --interpreter python3 --name V1 -- --actual_batch_size 6 --wallet.name Alice --wallet.hotkey default --bucket $BUCKET --device cuda:0 --use_wandb --project $PROJECT
pm2 start miner.py --interpreter python3 --name M1 -- --actual_batch_size 6 --wallet.name Alice --wallet.hotkey M1 --bucket $BUCKET --device cuda:1 --use_wandb --project $PROJECT
pm2 start miner.py --interpreter python3 --name M2 -- --actual_batch_size 6 --wallet.name Alice --wallet.hotkey M2 --bucket $BUCKET --device cuda:2 --use_wandb --project $PROJECT
pm2 start miner.py --interpreter python3 --name M3 -- --actual_batch_size 6 --wallet.name Alice --wallet.hotkey M3 --bucket $BUCKET --device cuda:3 --use_wandb --project $PROJECT
pm2 start miner.py --interpreter python3 --name M4 -- --actual_batch_size 6 --wallet.name Alice --wallet.hotkey M4 --bucket $BUCKET --device cuda:5 --use_wandb --random --project $PROJECT
pm2 start miner.py --interpreter python3 --name M5 -- --actual_batch_size 6 --wallet.name Alice --wallet.hotkey M5 --bucket $BUCKET --device cuda:6 --use_wandb --random --project $PROJECT
pm2 start miner.py --interpreter python3 --name M6 -- --actual_batch_size 6 --wallet.name Alice --wallet.hotkey M3 --bucket $BUCKET --device cuda:4 --use_wandb --baseline --project $PROJECT



