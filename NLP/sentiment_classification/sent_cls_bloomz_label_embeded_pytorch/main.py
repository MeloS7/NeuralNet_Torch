import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig
from datasets import load_dataset

from src.utils import load_config, train_function
from src.dataset import SST2Dataset


def main():
    # Read config file
    config = load_config("config.yaml")

    # Training parameters
    train_device = config["train"]["device"]
    learning_rate = float(config["train"]["lr"])
    num_epochs = int(config["train"]["num_epochs"])
    train_batch_size = int(config["train"]["train_batch_size"])

    # Evaluation parameters
    # eval_devide = config['eval']['device']
    eval_batch_size = int(config["eval"]["eval_batch_size"])

    # Model parameters
    model_name = config["model"]["model_name"]
    tokenizer_name = config["model"]["tokenizer_name"]
    max_length = config["model"]["max_length"]

    # PEFT parameters
    task_type = config["peft"]["task_type"]
    num_virtual_tokens = config["peft"]["num_virtual_tokens"]
    prompt_init_text = config["peft"]["prompt_init_text"]
    peft_config = PromptTuningConfig(
        task_type=task_type,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init_text=prompt_init_text,
        tokenizer_name_or_path=tokenizer_name,
    )

    # Dataset parameters
    dataset_name = config["dataset"]["dataset_name"]
    task_name = config["dataset"]["task_name"]
    train_ratio = config["dataset"]["train_ratio"]

    # Other parameters
    checkpoint_name = f"{dataset_name}_{model_name}_{task_name}_{num_epochs}".replace(
        "/", "_"
    )
    model_dir = os.path.join("res", "models")
    checkpoint_path = os.path.join(model_dir, checkpoint_name)

    # Load dataset
    dataset = load_dataset(dataset_name, task_name)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Preprocess dataset
    print("========Preprocessing dataset========")
    sst2dataset = SST2Dataset(
        tokenizer,
        max_length,
        dataset,
        task_name,
        train_ratio,
        train_batch_size,
        eval_batch_size,
    )

    # DataLoaders
    train_dataloader, eval_dataloader = sst2dataset.get_train_eval_dataloader()
    # test_dataloader = sst2dataset.get_test_dataloader()

    # Optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    print("========Training started========")
    trained_model = train_function(
        model,
        train_device,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        num_epochs,
        checkpoint_path,
    )
    print("========Training completed========")


if __name__ == "__main__":
    main()
