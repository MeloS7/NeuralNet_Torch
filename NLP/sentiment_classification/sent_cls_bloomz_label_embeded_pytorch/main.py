import yaml
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

from .src.utils import load_config

def main():
    # Read config file
    config = load_config('config.yaml')

    db_host = config['database']['host']
    db_port = config['database']['port']
    db_username = config['database']['username']
    db_password = config['database']['password']

    logging_level = config['logging']['level']
    logging_file_path = config['logging']['file_path']

    api_keys = config['api_keys']


if __name__ == "__main__":
    main()
