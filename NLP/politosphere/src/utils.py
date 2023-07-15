import yaml
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_function(model, device, train_dataloader, eval_dataloader, optimizer, lr_scheduler, num_epochs, checkpoints_path):
    # training and evaluation
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        # test generation
        # preds, labels = evaluate_function(model, device, eval_dataloader, tokenizer)
        # accuracy = calculate_percentage(get_metrics(preds, labels))
        # print(f"Epoch: {epoch} Accuracy: {accuracy}")
    
    # torch.save(model, checkpoints_path)
    model.save_pretrained(checkpoints_path)

def get_metrics(preds, labels):
    return [int(pred == label) for pred, label in zip(preds, labels)]

def calculate_percentage(lst):
    total = len(lst)
    count_ones = sum(lst)
    percentage = count_ones / total * 100
    return percentage

def evaluate_function(model, device, test_dataloader, tokenizer):
    model.eval()
    model = model.to(device)

    test_preds = []
    labels = []

    for step, batch in enumerate(tqdm(test_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], max_new_tokens=1, eos_token_id=3
            )
        # print(tokenizer.batch_decode(outputs[:, -10:-1]))
        print(tokenizer.batch_decode(outputs[:, -1]))
        print(tokenizer.batch_decode(batch["labels"]))

        ### For test if there are invalid characters
        # decoded_outputs = tokenizer.batch_decode(outputs[:, -1])
        # allowed_chars = ['0', '2', '3']
        # invalid_chars = [char for char in decoded_outputs if char not in allowed_chars]
        # assert 1==2
        # assert len(invalid_chars) == 0, f"Invalid characters found: {tokenizer.batch_decode(outputs)}"

        test_preds.extend(tokenizer.batch_decode(outputs[:, -1]))
        labels.extend(tokenizer.batch_decode(batch["labels"]))

    return test_preds, labels