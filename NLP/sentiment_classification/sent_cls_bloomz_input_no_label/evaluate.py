import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from src.utils import load_config, evaluate_function, get_metrics, calculate_percentage
from src.dataset import SST2Dataset


def evaluate():
    # Read config file
    config = load_config("config.yaml")

    # Dataset parameters
    dataset_name = config["dataset"]["dataset_name"]
    task_name = config["dataset"]["task_name"]

    # Model parameters
    model_name = config["model"]["model_name"]
    tokenizer_name = config["model"]["tokenizer_name"]
    max_length = config["model"]["max_length"]

    # Evaluation parameters
    eval_device = config["eval"]["device"]
    eval_batch_size = int(config["eval"]["eval_batch_size"])
    eval_model_path = config["eval"]["eval_model_path"]
    eval_base = config["eval"]["eval_base"]

    # Load dataset
    dataset = load_dataset(dataset_name, task_name)

    # Load model
    if eval_base:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        config = PeftConfig.from_pretrained(eval_model_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, eval_model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Preprocess dataset
    print("========Preprocessing dataset========")
    sst2dataset = SST2Dataset(
        tokenizer, max_length, dataset, task_name, eval_batch_size=eval_batch_size
    )

    # Load test dataloader
    test_dataloader = sst2dataset.get_test_dataloader()

    # Evaluate model
    print("========Evaluating started========")
    preds, labels = evaluate_function(
        model,
        eval_device,
        test_dataloader,
        tokenizer,
    )
    print("========Evaluating completed========")

    accuracy = calculate_percentage(get_metrics(preds, labels))
    print(f"Accuracy: {accuracy}")


# Run evaluate function
evaluate()
