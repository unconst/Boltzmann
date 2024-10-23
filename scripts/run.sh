#!/usr/bin/env bash

# The MIT License (MIT)
# © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# [License continues...]

set -euo pipefail

# Initialize default values
DEBUG=false
PROJECT="aesop"
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
BUCKET=""

# Function to display help message
display_help() {
    cat << EOF
Usage: $0 [options]

Options:
    --debug                     Enable debug mode
    --project <project_name>    Set the project name (default: aesop)
    --aws-access-key-id <key>   Set AWS Access Key ID
    --aws-secret-access-key <key> Set AWS Secret Access Key
    --bucket <bucket_name>      Set the S3 bucket name
    -h, --help                  Display this help message

Description:
    Installs and runs a Boltzmann miner on your GPU.
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --debug)
            DEBUG=true
            shift
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --aws-access-key-id)
            AWS_ACCESS_KEY_ID="$2"
            shift 2
            ;;
        --aws-secret-access-key)
            AWS_SECRET_ACCESS_KEY="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        -h|--help|-help|--h)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

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

# Logging functions
ohai() {
    printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$*"
}

pdone() {
    printf "  ${tty_green}[✔]${tty_bold} %s${tty_reset}\n" "$*"
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

trap 'abort "An unexpected error occurred."' ERR

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

have_sudo_access() {
    if ! command -v sudo &> /dev/null; then
        warn "sudo command not found. Please install sudo or run as root."
        return 1
    fi
    if [ "$EUID" -ne 0 ]; then 
        if ! sudo -n true 2>/dev/null; then
            warn "This script requires sudo access to install packages. Please run as root or ensure your user has sudo privileges."
            return 1
        fi
    fi
    return 0
}

execute_sudo() {
    if have_sudo_access; then
        ohai "sudo $*"
        if ! sudo "$@"; then
            abort "Failed to execute: sudo $*"
        fi
    else
        warn "Sudo access is required, attempting to run without sudo"
        ohai "$*"
        if ! "$@"; then
            abort "Failed to execute: $*"
        fi
    fi
}

# Function to set or replace environment variables in bash_profile
set_or_replace_env_var() {
    local var_name="$1"
    local var_value="$2"
    local profile_file="$3"

    # Escape special characters for sed
    local escaped_var_value=$(printf '%s\n' "$var_value" | sed -e 's/[\/&]/\\&/g')

    if grep -q "^export $var_name=" "$profile_file"; then
        # Variable exists, replace it
        sed -i.bak "s/^export $var_name=.*/export $var_name=\"$escaped_var_value\"/" "$profile_file"
    else
        # Variable does not exist, append it
        echo "export $var_name=\"$var_value\"" >> "$profile_file"
    fi
}

# Clear the screen and display the logo
clear
echo ""
echo ""
echo " ______   _____         _______ ______ _______ _______ __   _ __   _"
echo " |_____] |     | |         |     ____/ |  |  | |_____| | \  | | \  |"
echo " |_____] |_____| |_____    |    /_____ |  |  | |     | |  \_| |  \_|"
echo "                                                                    "
echo ""
echo ""

echo "This script will do the following:"
echo "1. Install required software (Git, npm, pm2, Python 3.12)"
echo "2. Set up AWS credentials"
echo "3. Clone and set up the Boltzmann repository"
echo "4. Create and register Bittensor wallets"
echo "5. Configure wandb for logging"
echo "6. Clean the specified S3 bucket"
echo "7. Start Boltzmann miners on available GPUs"
echo ""
echo "Please ensure you have a stable internet connection and sufficient permissions to install software."
echo ""

wait_for_user

# Ensure ~/.bash_profile exists
touch ~/.bash_profile
source ~/.bash_profile

# Backup the bash_profile
cp ~/.bash_profile ~/.bash_profile.bak

# Prompt the user for AWS credentials if not supplied via command-line
ohai "Getting AWS credentials ..."
if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]] || [[ -z "$BUCKET" ]]; then
    # TODO: Consider securely storing AWS credentials rather than storing them in plain text
    warn "This script will store your AWS credentials in your ~/.bash_profile file."
    warn "This is not secure and is not recommended."
    read -p "Do you want to proceed? [y/N]: " proceed
    if [[ "$proceed" != "y" && "$proceed" != "Y" ]]; then
        abort "Aborted by user."
    fi

    if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
        read -p "Enter your AWS Access Key ID: " AWS_ACCESS_KEY_ID
    fi
    if [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
        read -p "Enter your AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    fi
    if [[ -z "$BUCKET" ]]; then
        read -p "Enter your S3 Bucket Name: " BUCKET
    fi
fi

# Overwrite or add the AWS credentials in the bash_profile
set_or_replace_env_var "AWS_ACCESS_KEY_ID" "$AWS_ACCESS_KEY_ID" ~/.bash_profile
set_or_replace_env_var "AWS_SECRET_ACCESS_KEY" "$AWS_SECRET_ACCESS_KEY" ~/.bash_profile
set_or_replace_env_var "BUCKET" "$BUCKET" ~/.bash_profile

# Source the bash_profile to apply the changes
source ~/.bash_profile
pdone "AWS credentials set in ~/.bash_profile"

ohai "Installing requirements ..."
# Install Git if not present
if ! command -v git &> /dev/null; then
    ohai "Git not found. Installing git ..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ohai "Detected Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                ohai "Detected Ubuntu, installing Git..."
                if [[ "$DEBUG" == "true" ]]; then
                    execute_sudo apt-get update -y
                    execute_sudo apt-get install git -y
                else
                    execute_sudo apt-get update -y > /dev/null 2>&1
                    execute_sudo apt-get install git -y > /dev/null 2>&1
                fi
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
        if ! command -v brew &> /dev/null; then
            warn "Homebrew is not installed, installing Homebrew..."
            execute /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        if [[ "$DEBUG" == "true" ]]; then
            execute brew install git
        else
            execute brew install git > /dev/null 2>&1
        fi
    else
        abort "Unsupported OS type: $OSTYPE"
    fi
else
    pdone "Git is already installed"
fi

# TODO: Add error handling for package installations
# TODO: Ensure compatibility with different package managers

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    ohai "Installing npm ..."
    if ! command -v node &> /dev/null; then
        ohai "Node.js could not be found, installing..."
        if ! curl -fsSL https://deb.nodesource.com/setup_18.x | bash; then
            abort "Failed to download Node.js setup script"
        fi
        if ! execute_sudo apt-get install -y nodejs; then
            abort "Failed to install Node.js"
        fi
    fi
    if ! curl -L https://www.npmjs.com/install.sh | sh; then
        abort "Failed to install npm"
    fi
fi
pdone "npm is installed"

# Install pm2
if ! command -v pm2 &> /dev/null; then
    ohai "Installing pm2 ..."
    if [[ "$DEBUG" == "true" ]]; then
        execute npm install pm2 -g
    else
        execute npm install pm2 -g > /dev/null 2>&1
    fi
fi
pdone "pm2 is installed"

ohai "Installing Boltzmann ..."
# Check if we are inside the boltzmann repository
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    REPO_PATH="."
else
    if [ ! -d "boltzmann" ]; then
        ohai "Cloning boltzmann ..."
        execute git clone https://github.com/unconst/boltzmann
        REPO_PATH="boltzmann/"
    else
        REPO_PATH="boltzmann/"
    fi
fi
pdone "Boltzmann repository is ready at $REPO_PATH"

# Install Python 3.12 if not installed
if ! command -v python3.12 &> /dev/null; then
    ohai "Installing python3.12 ..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ohai "Detected Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID_LIKE" == *"ubuntu"* ]]; then
                ohai "Detected Ubuntu, installing Python 3.12..."
                if [[ "$DEBUG" == "true" ]]; then
                    if have_sudo_access; then
                        execute_sudo add-apt-repository ppa:deadsnakes/ppa -y
                    else
                        warn "Skipping add-apt-repository due to lack of sudo access"
                    fi
                    execute_sudo apt-get update -y
                else
                    if have_sudo_access; then
                        execute_sudo add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
                    else
                        warn "Skipping add-apt-repository due to lack of sudo access"
                    fi
                    execute_sudo apt-get update -y > /dev/null 2>&1
                    execute_sudo apt-get install --reinstall python3-apt > /dev/null 2>&1
                    execute_sudo apt-get install python3.12 -y > /dev/null 2>&1
                    execute_sudo apt-get install python3.12-venv > /dev/null 2>&1
                fi
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
        if ! command -v brew &> /dev/null; then
            warn "Homebrew is not installed, installing Homebrew..."
            execute /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        if [[ "$DEBUG" == "true" ]]; then
            execute brew install python@3.12
        else
            execute brew install python@3.12 > /dev/null 2>&1
        fi
    else
        abort "Unsupported OS type: $OSTYPE"
    fi
fi
pdone "Python 3.12 is installed"

# Create a virtual environment if it does not exist
if [ ! -d "$REPO_PATH/venv" ]; then
    ohai "Creating virtual environment at $REPO_PATH..."
    if [[ "$DEBUG" == "true" ]]; then
        execute python3.12 -m venv "$REPO_PATH/venv"
    else
        execute python3.12 -m venv "$REPO_PATH/venv" > /dev/null 2>&1
    fi
fi
pdone "Virtual environment is set up at $REPO_PATH"

# Activate the virtual environment
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ "$DEBUG" == "true" ]]; then
        source $REPO_PATH/venv/bin/activate
    else
        source $REPO_PATH/venv/bin/activate > /dev/null 2>&1
    fi
fi
pdone "Virtual environment activated"

ohai "Installing Python requirements ..."
if [[ "$DEBUG" == "true" ]]; then
    execute pip install -r $REPO_PATH/requirements.txt
    execute pip install --upgrade cryptography pyOpenSSL
else
    execute pip install -r $REPO_PATH/requirements.txt > /dev/null 2>&1
    execute pip install --upgrade cryptography pyOpenSSL > /dev/null 2>&1
fi
pdone "Python requirements installed"

# Check for GPUs
ohai "Checking for GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    warn "nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
    NUM_GPUS=0
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

    if [ "$NUM_GPUS" -gt 0 ]; then
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | while read -r memory; do
            pdone "Found GPU with $((memory / 1024)) GB of memory"
        done
    else
        warn "No GPUs found on this machine."
    fi
fi

# Check system RAM
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    pdone "System RAM: ${TOTAL_RAM} GB"
else
    warn "Cannot determine system RAM. 'free' command not found."
fi

ohai "Creating wallets ..."
# Create the default key
if ! python3 -c "import bittensor as bt; w = bt.wallet(); print(w.coldkey_file.exists_on_device())" | grep -q "True"; then
    execute btcli w new_coldkey --wallet.path ~/.bittensor/wallets --wallet.name default --n-words 12 
fi
pdone "Wallet 'default' is ready"

# Ensure btcli is installed
if ! command -v btcli &> /dev/null; then
    abort "btcli command not found. Please ensure it is installed."
fi

# Create hotkeys and register them
if [ "$NUM_GPUS" -gt 0 ]; then
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        # Check if the hotkey file exists on the device
        exists_on_device=$(python3 -c "import bittensor as bt; w = bt.wallet(hotkey='C$i'); print(w.hotkey_file.exists_on_device())" 2>/dev/null)
        if [ "$exists_on_device" != "True" ]; then
            echo "n" | btcli wallet new_hotkey --wallet.name default --wallet.hotkey C$i --n-words 12 > /dev/null 2>&1;
        fi
        pdone "Created Hotkey 'C$i'"

        # Check if the hotkey is registered on subnet 220
        is_registered=$(python3 -c "import bittensor as bt; w = bt.wallet(hotkey='C$i'); sub = bt.subtensor('test'); print(sub.is_hotkey_registered_on_subnet(hotkey_ss58=w.hotkey.ss58_address, netuid=220))" 2>/dev/null)
        if [[ "$is_registered" != *"True"* ]]; then
            ohai "Registering hotkey 'C$i' on subnet 220"
            btcli subnet pow_register --wallet.name default --wallet.hotkey C$i --netuid 220 --subtensor.network test --no_prompt > /dev/null 2>&1;
        fi
        pdone "Registered Hotkey 'C$i' on subnet 220"
    done
else
    warn "No GPUs found. Skipping hotkey creation."
    exit
fi
pdone "All hotkeys registered"

ohai "Logging into wandb..."
execute wandb login
pdone "wandb is configured"

# Clean the bucket
ohai "Cleaning bucket $BUCKET..."
if [[ "$DEBUG" == "true" ]]; then
    execute python3 $REPO_PATH/tools/clean.py --bucket "$BUCKET"
else
    execute python3 $REPO_PATH/tools/clean.py --bucket "$BUCKET" > /dev/null 2>&1
fi
pdone "Bucket '$BUCKET' cleaned"

# Close down all previous processes and restart them
if pm2 list | grep -q 'online'; then
    ohai "Stopping old pm2 processes..."
    pm2 delete all
    pdone "Old processes stopped"
fi

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
        if [ "$GPU_MEMORY" -ge 80000 ]; then
            BATCH_SIZE=6
        elif [ "$GPU_MEMORY" -ge 40000 ]; then
            BATCH_SIZE=3
        elif [ "$GPU_MEMORY" -ge 20000 ]; then
            BATCH_SIZE=1
        else
            BATCH_SIZE=1
        fi
        ohai "Starting miner on GPU $GPU_INDEX with batch size $BATCH_SIZE..."
        if [[ "$DEBUG" == "true" ]]; then
            execute pm2 start "$REPO_PATH/miner.py" --interpreter python3 --name C$i -- --actual_batch_size "$BATCH_SIZE" --wallet.name default --wallet.hotkey C$i --bucket "$BUCKET" --device cuda:$GPU_INDEX --use_wandb --project "$PROJECT"
        else
            execute pm2 start "$REPO_PATH/miner.py" --interpreter python3 --name C$i -- --actual_batch_size "$BATCH_SIZE" --wallet.name default --wallet.hotkey C$i --bucket "$BUCKET" --device cuda:$GPU_INDEX --use_wandb --project "$PROJECT" > /dev/null 2>&1
        fi
    done
else
    warn "No GPUs found. Skipping miner startup."
fi
pdone "All miners started"
pm2 list

echo ""
pdone "SUCCESS"
echo ""

# Start logging process 1
pm2 logs C0

# TODO: Implement proper cleanup and error handling
# TODO: Add checks for all dependencies and versions
# TODO: Securely handle AWS credentials and consider alternatives to storing them in plain text
