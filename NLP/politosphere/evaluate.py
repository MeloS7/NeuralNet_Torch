import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from src.utils import load_config, evaluate_function, get_metrics, calculate_percentage
from src.dataset import PolitosphereDataset


def evaluate():
    # Read config file
    config = load_config("config.yaml")

    # Dataset parameters
    dataset_path = config["dataset"]["dataset_path"]
    eval_dataset_path = config["dataset"]["eval_dataset_path"]
    test_dataset_path = config["dataset"]["test_dataset_path"]

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
    dataset = load_dataset("json", data_files=dataset_path)
    eval_dataset = load_dataset("json", data_files=eval_dataset_path)
    test_dataset = load_dataset("json", data_files=test_dataset_path)

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
    sst2dataset = PolitosphereDataset(
        tokenizer, max_length, dataset, eval_dataset, test_dataset, eval_batch_size=eval_batch_size
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
