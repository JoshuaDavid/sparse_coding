import json
import os
import subprocess
import sys
from typing import Optional, Union, List
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

VAST_NUM = 4
# DEST_ADDR = f"root@ssh{VAST_NUM}.vast.ai"
DEST_ADDR = "mchorse@216.153.50.63"
SSH_PYTHON = "/opt/conda/bin/python"

PORT = 22

USER = "aidan"

SSH_DIRECTORY = f"sparse_coding_{USER}"
BUCKET_NAME = "sparse-coding"

ACCESS_KEY_NAME_DICT = {
    "AKIAV3IKT32M2ZA3WRLQ": "hoagy",
    "AKIATUSYDLZAEZ7T5GLX": "aidan",
    "AKIATEQID7TUM5FUW4R5": "logan",
}

def sync():
    """Sync the local directory with the remote host."""
    command = f'rsync -rv --filter ":- .gitignore" --exclude ".git" -e "ssh -p {PORT}" . {DEST_ADDR}:{SSH_DIRECTORY}'
    subprocess.call(command, shell=True)

def autointerp_sync():
    """Sync the local directory with the remote host's auto interp results, excluding hdf files."""
    command = f'rsync -r --exclude "*.hdf" --exclude "*.pkl" -e ssh {DEST_ADDR}:{SSH_DIRECTORY}/auto_interp_results . '
    print(command)
    subprocess.call(command, shell=True)

def copy_models():
    """Copy the models from local directory to the remote host."""
    command = f"scp -P {PORT} -r models {DEST_ADDR}:{SSH_DIRECTORY}/models"
    subprocess.call(command, shell=True)
    # also copying across a few other files
    command = f"scp -P {PORT} -r outputs/thinrun/autoencoders_cpu.pkl {DEST_ADDR}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)

def copy_secrets():
    """Copy the secrets.json file from local directory to the remote host."""
    command = f"scp -P {PORT} secrets.json {DEST_ADDR}:{SSH_DIRECTORY}"
    subprocess.call(command, shell=True)


def copy_recent():
    """Get the most recent outputs folder in the remote host and copy across to same place in local directory."""
    # get the most recent folders
    command = f'ssh -p {PORT} {DEST_ADDR} "ls -td {SSH_DIRECTORY}/outputs/* | head -1"'
    output = subprocess.check_output(command, shell=True)
    output = output.decode("utf-8").strip()
    # copy across
    command = f"scp -P {PORT} -r {DEST_ADDR}:{output} outputs"
    subprocess.call(command, shell=True)

def copy_dotfiles():
    """Copy dotfiles into remote host and run install and deploy scripts"""
    df_dir = f"dotfiles_{USER}"
    command = f"scp -P {PORT} -r ~/git/dotfiles {DEST_ADDR}:{df_dir}"
    subprocess.call(command, shell=True)
    command = f"ssh -p {PORT} {DEST_ADDR} 'cd ~/{df_dir} && ./install.sh && ./deploy.sh'"
    subprocess.call(command, shell=True)

def setup():
    """Sync, copy models, create venv and install requirements."""
    sync()
    copy_models()
    copy_secrets()
    command = f'ssh -p {PORT} {DEST_ADDR} "cd {SSH_DIRECTORY} && sudo apt -y install python3.9 python3.9-venv && python3.9 -m venv .env --system-site-packages && source .env/bin/activate && pip install -r requirements.txt" && apt install vim'
    # command = f"ssh -p {VAST_PORT} {dest_addr} \"cd {SSH_DIRECTORY} && echo $PATH\""
    subprocess.call(command, shell=True)

import warnings

class dotdict(dict):
    """Dictionary that can be accessed with dot notation."""

    def __init__(self, d: Optional[dict] = None):
        if d is None:
            d = {}
        super().__init__(d)

    def __dict__(self):
        return self

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Attribute {name} not found")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def check_use_baukit(model_name):
    if model_name in ["nanoGPT"]:
        return True
    elif model_name in ["gpt2", "EleutherAI/pythia-70m-deduped", "roneneldan/TinyStories-33M"]:
        return False
    else:
        raise NotImplementedError(f"Unknown if model {model_name} uses baukit")


def get_activation_size(model_name: str, layer_loc: str):
    if model_name == "EleutherAI/pythia-70m-deduped":
        residual_dim = 512
    elif model_name in ["EleutherAI/pythia-160m-deduped", "gpt2", "roneneldan/TinyStories-33M"]:
        residual_dim = 768
    elif model_name == "nanoGPT":
        residual_dim = 32

    if layer_loc == "mlp":
        return residual_dim * 4
    else:
        return residual_dim
    

def make_tensor_name(layer: int, layer_loc: str, model_name: str) -> str:
    """Make the tensor name for a given layer and model."""
    assert layer_loc in ["residual", "mlp", "attn", "attn_k", "mlp_out"], f"Layer location {layer_loc} not supported"
    if layer_loc == "residual":
        if model_name in ["gpt2", "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-160m-deduped", "roneneldan/TinyStories-33M"]:
            tensor_name = f"blocks.{layer}.hook_resid_post"
        else:
            raise NotImplementedError(f"Model {model_name} not supported for residual stream")
    elif layer_loc == "mlp":
        if model_name in ["gpt2", "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-160m-deduped", "roneneldan/TinyStories-33M"]:
            tensor_name = f"blocks.{layer}.mlp.hook_post"
        elif model_name == "nanoGPT":
            tensor_name = f"transformer.h.{layer}.mlp.c_fc"
        else:
            raise NotImplementedError(f"Model {model_name} not supported for MLP")
    elif layer_loc == "attn":
        if model_name in ["gpt2", "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-160m-deduped", "roneneldan/TinyStories-33M"]:
            tensor_name = f"blocks.{layer}.hook_resid_post"
        else:
            raise NotImplementedError(f"Model {model_name} not supported for attention stream")
    elif layer_loc == "attn_k":
        if model_name in ["gpt2", "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-160m-deduped", "roneneldan/TinyStories-33M"]:
            tensor_name = f"blocks.{layer}.attn.hook_k"
        else:
            raise NotImplementedError(f"Model {model_name} not supported for attention stream")
    elif layer_loc == "mlp_out":
        if model_name in ["gpt2", "EleutherAI/pythia-70m-deduped", "EleutherAI/pythia-160m-deduped", "roneneldan/TinyStories-33M"]:
            tensor_name = f"blocks.{layer}.hook_mlp_out"
        else:
            raise NotImplementedError(f"Model {model_name} not supported for MLP")

    return tensor_name

def upload_to_aws(local_file_name) -> bool:
    """"
    Upload a file to an S3 bucket
    :param local_file_name: File to upload
    :param s3_file_name: S3 object name. If not specified then local_file_name is used
    """
    secrets = json.load(open("secrets.json"))

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )

    if secrets["access_key"] in ACCESS_KEY_NAME_DICT:
        name = ACCESS_KEY_NAME_DICT[secrets["access_key"]]
    else:
        name = "unknown"

    s3_file_name = name + "-" + local_file_name

    local_file_path = Path(local_file_name)
    try:
        if local_file_path.is_dir():
            _upload_directory(local_file_name, s3)
        else:
            s3.upload_file(str(local_file_name), BUCKET_NAME, str(s3_file_name))
        print(f"Upload Successful of {local_file_name}")
        return True
    except FileNotFoundError:
        print(f"File {local_file_name} was not found")
        return False
    except NoCredentialsError: # mypy: ignore, not sure why it doesn't think it's a valid exception class
        print("Credentials not available")
        return False
    
def _upload_directory(path, s3_client):
    for root, dirs, files in os.walk(path):
        for file_name in files:
            full_file_name = os.path.join(root, file_name)
            s3_client.upload_file(str(full_file_name), BUCKET_NAME, str(full_file_name))

def download_from_aws(files: Union[str, List[str]], force_redownload: bool = False) -> bool:
    """
    Download a file from an S3 bucket
    :param files: List of files to download
    :param force_redownload: If True, will download even if the file already exists
    
    Returns:
        True if all files were downloaded successfully, False otherwise
    """
    secrets = json.load(open("secrets.json"))
    if isinstance(files, str):
        files = [files]
    
    if not force_redownload:
        files = [f for f in files if not os.path.exists(f)]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["access_key"],
        aws_secret_access_key=secrets["secret_key"],
    )
    all_correct = True
    for filename in files:
        try:
            parent_dir = os.path.dirname(filename)
            if not os.path.exists(parent_dir) and parent_dir != "":
                os.makedirs(os.path.dirname(filename))
            with open(filename, "wb") as f:
                s3.download_fileobj(BUCKET_NAME, filename, f)

            print(f"Successfully downloaded file: {filename}")
        except ClientError:
            print(f"File: {filename} does not exist")
            all_correct = False

    return all_correct
    

if __name__ == "__main__":
    if sys.argv[1] == "sync":
        sync()
    elif sys.argv[1] == "models":
        copy_models()
    elif sys.argv[1] == "recent":
        copy_recent()
    elif sys.argv[1] == "setup":
        setup()
    elif sys.argv[1] == "secrets":
        copy_secrets()
    elif sys.argv[1] == "interp_sync":
        autointerp_sync()
    elif sys.argv[1] == "dotfiles":
        copy_dotfiles()
    else:
        raise NotImplementedError(f"Command {sys.argv[1]} not recognised")
