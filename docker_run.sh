#!/usr/bin/env bash

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

set -euo pipefail

trap 'abort "An unexpected error occurred."' ERR

# Set up colors and styles
if [[ -t 1 ]]; then
    tty_escape() { printf "\033[%sm" "$1"; }
else
    tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_green="$(tty_mkbold 32)"
tty_yellow="$(tty_mkbold 33)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

ohai() {
    printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$*"
}

pdone() {
    printf "${tty_green}[✔]${tty_bold} %s${tty_reset}\n" "$*"
}

info() {
    printf "${tty_green}%s${tty_reset}\n" "$*"
}

warn() {
    printf "${tty_yellow}Warning${tty_reset}: %s\n" "$*" >&2
}

error() {
    printf "${tty_red}Error${tty_reset}: %s\n" "$*" >&2
}

abort() {
    error "$@"
    exit 1
}

getc() {
  local save_state
  save_state="$(/bin/stty -g)"
  /bin/stty raw -echo
  IFS='' read -r -n 1 -d '' "$@"
  /bin/stty "${save_state}"
}

wait_for_user() {
  local c
  echo
  echo "Press ${tty_bold}RETURN${tty_reset}/${tty_bold}ENTER${tty_reset} to continue or any other key to abort:"
  getc c
  # we test for \r and \n because some stuff does \r instead
  if ! [[ "${c}" == $'\r' || "${c}" == $'\n' ]]
  then
    exit 1
  fi
}

execute() {
    ohai "Running: $*"
    if ! "$@"; then
        abort "Failed during: $*"
    fi
}

have_root_access() {
    if [ "$EUID" -ne 0 ]; then
        warn "This script requires root privileges to install packages."
        return 1
    fi
    return 0
}

clear
echo ""
echo ""
echo " ______   _____         _______ ______ _______ _______ __   _ __   _"
echo " |_____] |     | |         |     ____/ |  |  | |_____| | \  | | \  |"
echo " |_____] |_____| |_____    |    /_____ |  |  | |     | |  \_| |  \_|"
echo "                                                                    "
echo ""
echo ""

wait_for_user

# Install Git if not present
if ! command -v git &> /dev/null; then
    ohai "Installing git ..."
    if have_root_access; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            ohai "Detected Linux"
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                    ohai "Detected Ubuntu, installing Git..."
                    execute apt-get update -y
                    execute apt-get install -y git
                else
                    warn "Unsupported Linux distribution: $ID"
                    abort "Cannot install Git automatically"
                fi
            else
                warn "Cannot detect Linux distribution"
                abort "Cannot install Git automatically"
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            ohai "Detected macOS, installing Git..."
            execute xcode-select --install
        else
            abort "Unsupported OS type: $OSTYPE"
        fi
    else
        abort "Root access is required to install Git."
    fi
else
    pdone "Found Git"
fi

# Check if we are inside the cont repository
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    REPO_PATH="."
else
    if [ ! -d "cont" ]; then
        ohai "Cloning boltzmann ..."
        execute git clone https://github.com/unconst/cont
        REPO_PATH="cont/"
    else
        REPO_PATH="cont/"
    fi
fi
pdone "Pulled Boltzmann $REPO_PATH"

# Install Node.js and npm if not present
if ! command -v npm &> /dev/null; then
    ohai "Installing Node.js and npm ..."
    if have_root_access; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            ohai "Detected Linux"
            execute apt-get update -y
            execute apt-get install -y curl
            curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
            execute apt-get install -y nodejs
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            ohai "Detected macOS, installing Node.js and npm..."
            execute brew install node
        else
            abort "Unsupported OS type: $OSTYPE"
        fi
    else
        abort "Root access is required to install Node.js and npm."
    fi
    pdone "Installed Node.js and npm"
else
    pdone "Found npm"
fi

# Install pm2
if ! command -v pm2 &> /dev/null; then
    ohai "Installing pm2 ..."
    execute npm install pm2 -g
    pdone "Installed pm2"
else
    pdone "Found pm2"
fi

# Install Python 3.12 if not installed
if ! command -v python3.12 &> /dev/null; then
    ohai "Installing python3.12 ..."
    if have_root_access; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            ohai "Detected Linux"
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                    ohai "Detected Ubuntu, installing Python 3.12..."
                    execute apt-get update -y
                    execute apt-get install -y software-properties-common gnupg

                    # Add the deadsnakes PPA manually
                    ohai "Adding deadsnakes PPA manually..."
                    execute mkdir -p /etc/apt/keyrings
                    execute curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x6A755776" | gpg --dearmor --batch --yes -o /etc/apt/keyrings/deadsnakes-archive-keyring.gpg
                    echo "deb [signed-by=/etc/apt/keyrings/deadsnakes-archive-keyring.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list

                    execute apt-get update -y
                    execute apt-get install -y python3.12 python3.12-venv

                else
                    warn "Unsupported Linux distribution: $ID"
                    abort "Cannot install Python 3.12 automatically"
                fi
            else
                warn "Cannot detect Linux distribution"
                abort "Cannot install Python 3.12 automatically"
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            ohai "Detected macOS, installing Python 3.12..."
            execute brew install python@3.12
        else
            abort "Unsupported OS type: $OSTYPE"
        fi
    else
        abort "Root access is required to install Python 3.12."
    fi
    pdone "Installed python3.12"
else
    pdone "Found python3.12"
fi

touch ~/.bash_profile

# Prompt the user for AWS credentials and inject them into the bash_profile file if not already stored
if ! grep -q "AWS_ACCESS_KEY_ID" ~/.bash_profile || ! grep -q "AWS_SECRET_ACCESS_KEY" ~/.bash_profile || ! grep -q "BUCKET" ~/.bash_profile; then
    clear
    warn "This script will store your AWS credentials in your ~/.bash_profile file."
    warn "This is not secure and is not recommended."
    read -p "Do you want to proceed? [y/N]: " proceed
    if [[ "$proceed" != "y" && "$proceed" != "Y" ]]; then
        abort "Aborted by user."
    fi

    read -p "Enter your AWS Access Key ID: " AWS_ACCESS_KEY_ID
    read -p "Enter your AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    read -p "Enter your S3 Bucket Name: " BUCKET

    echo "export AWS_ACCESS_KEY_ID=\"$AWS_ACCESS_KEY_ID\"" >> ~/.bash_profile
    echo "export AWS_SECRET_ACCESS_KEY=\"$AWS_SECRET_ACCESS_KEY\"" >> ~/.bash_profile
    echo "export BUCKET=\"$BUCKET\"" >> ~/.bash_profile
fi

# Source the bash_profile file to apply the changes
source ~/.bash_profile

pdone "Found AWS credentials"

# Create a virtual environment if it does not exist
if [ ! -d "$REPO_PATH/venv" ]; then
    ohai "Creating virtual environment at $REPO_PATH..."
    execute python3.12 -m venv "$REPO_PATH/venv"
fi
pdone "Created venv at $REPO_PATH"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    ohai "Activating virtual environment..."
    source "$REPO_PATH/venv/bin/activate"
fi
pdone "Activated venv at $REPO_PATH"

ohai "Installing requirements..."
execute pip install --upgrade pip
execute pip install -r "$REPO_PATH/requirements.txt"
pdone "Installed requirements"

# Check for GPUs
ohai "Checking for GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
    NUM_GPUS=0
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    ohai "Number of GPUs: $NUM_GPUS"

    if [ "$NUM_GPUS" -gt 0 ]; then
        ohai "GPU Memory Information:"
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | while read -r memory; do
            ohai "$((memory / 1024)) GB"
        done
    else
        warn "No GPUs found on this machine."
    fi
fi

# Check system RAM
ohai "Checking system RAM..."
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    ohai "Total System RAM: $TOTAL_RAM GB"
else
    warn "Cannot determine system RAM. 'free' command not found."
fi

# Create the default key
ohai "Creating the coldkey"
if ! python3.12 -c "import bittensor as bt; w = bt.wallet(); print(w.coldkey_file.exists_on_device())" | grep -q "True"; then
    execute btcli w new_coldkey --wallet.path ~/.bittensor/wallets --wallet.name default --n-words 12 --no_password
else
    ohai "Default key already exists on device."
fi

# Ensure btcli is installed
if ! command -v btcli &> /dev/null; then
    abort "btcli command not found. Please ensure it is installed."
fi

# Create hotkeys and register them
if [ "$NUM_GPUS" -gt 0 ]; then
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        # Check if the hotkey file exists on the device
        exists_on_device=$(python3.12 -c "import bittensor as bt; w = bt.wallet(hotkey='C$i'); print(w.hotkey_file.exists_on_device())" 2>/dev/null)
        if [ "$exists_on_device" != "True" ]; then
            echo "n" | btcli wallet new_hotkey --wallet.name default --wallet.hotkey C$i --n-words 12 > /dev/null 2>&1;
        else
            ohai "Hotkey C$i already exists on device."
        fi

        # Check if the hotkey is registered on subnet 220
        is_registered=$(python3.12 -c "import bittensor as bt; w = bt.wallet(hotkey='C$i'); sub = bt.subtensor('test'); print(sub.is_hotkey_registered_on_subnet(hotkey_ss58=w.hotkey.ss58_address, netuid=220))" 2>/dev/null)
        if [[ "$is_registered" != *"True"* ]]; then
            ohai "Registering key on subnet 220"
            btcli subnet pow_register --wallet.name default --wallet.hotkey C$i --netuid 220 --subtensor.network test --no_prompt > /dev/null 2>&1;
        else
            ohai "Key is already registered on subnet 220"
        fi
    done
else
    warn "No GPUs found. Skipping hotkey creation."
fi

ohai "Logging into wandb..."
execute wandb login

# Delete items from bucket
PROJECT=${2:-aesop}
ohai "Cleaning bucket $BUCKET..."
execute python3.12 "$REPO_PATH/tools/clean.py" --bucket "$BUCKET"

# Start all the processes again
if [ "$NUM_GPUS" -gt 0 ]; then
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        # Adjust GPU index for zero-based numbering
        GPU_INDEX=$i
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "$((i + 1))p")
        if [ -z "$GPU_MEMORY" ]; then
            warn "Could not get GPU memory for GPU $i"
            continue
        fi
        # Determine batch size based on GPU memory
        # This section adjusts the batch size for the miner based on the available GPU memory
        # Higher memory allows for larger batch sizes, which can improve performance
        if [ "$GPU_MEMORY" -ge 80000 ]; then
            # For GPUs with 80GB or more memory, use a batch size of 6
            BATCH_SIZE=6
        elif [ "$GPU_MEMORY" -ge 40000 ]; then
            # For GPUs with 40GB to 79GB memory, use a batch size of 3
            BATCH_SIZE=3
        elif [ "$GPU_MEMORY" -ge 20000 ]; then
            # For GPUs with 20GB to 39GB memory, use a batch size of 1
            BATCH_SIZE=1
        else
            # For GPUs with less than 20GB memory, also use a batch size of 1
            # This ensures that even lower-end GPUs can still participate
            BATCH_SIZE=1
        fi
        ohai "Starting miner on GPU $GPU_INDEX with batch size $BATCH_SIZE..."
        execute pm2 start "$REPO_PATH/miner.py" --interpreter python3 --name C$i -- --actual_batch_size "$BATCH_SIZE" --wallet.name default --wallet.hotkey C$i --bucket "$BUCKET" --device cuda:$GPU_INDEX --use_wandb --project "$PROJECT"
    done
else
    warn "No GPUs found. Skipping miner startup."
fi

pm2 list

ohai "Script completed successfully."
