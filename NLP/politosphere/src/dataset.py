import torch
from torch.utils.data import random_split, DataLoader
from transformers import default_data_collator


class SST2Dataset:
    def __init__(
        self,
        tokenizer,
        max_length,
        dataset,
        task_name,
        train_ratio=0.1,
        train_batch_size=8,
        eval_batch_size=8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = dataset
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.train_batch_size = int(train_batch_size)
        self.eval_batch_size = eval_batch_size
        self.text_column = "sentence"
        self.label_column = "text_label"

    def create_text_label(self):
        """Create text_label from label"""
        classes = [k for k in self.dataset["train"].features["label"].names]
        dataset = self.dataset.map(
            lambda x: {"text_label": [classes[label] for label in x["label"]]},
            batched=True,
            num_proc=1,
        )
        self.dataset = dataset
        return dataset

    def preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])

        # Embed the labels into the prompt
        inputs = [
            f"{self.text_column} : {x} Label : " for x in examples[self.text_column]
        ]
        targets = [str(x) for x in examples[self.label_column]]

        # First tokenize the inputs
        model_inputs = self.tokenizer(inputs)
        labels = self.tokenizer(targets)

        # Concatenate the inputs and labels and set -100 for labels
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            # Why Add eos_token_id at the end of the label for showing end?
            label_input_ids = labels["input_ids"][i] + [self.tokenizer.pad_token_id]

            # Concatenate the inputs and labels and set -100 for labels
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            # Set all attention mask to 1
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        # Pad the inputs and labels
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                self.max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (
                self.max_length - len(sample_input_ids)
            ) + label_input_ids

            # Truncate the inputs and labels
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][: self.max_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][: self.max_length]
            )
            labels["input_ids"][i] = torch.tensor(
                labels["input_ids"][i][: self.max_length]
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_train_eval_dataloader(self):
        dataset = self.create_text_label()
        processed_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        torch.manual_seed(42)
        train_len = len(train_dataset)
        ratio_len = int(train_len * self.train_ratio)
        print(f"Number of Training Examples: {ratio_len}")

        subset_train_dataset, _ = random_split(
            train_dataset, [ratio_len, train_len - ratio_len]
        )

        train_dataloader = DataLoader(
            subset_train_dataset,
            shuffle=True,
            collate_fn=default_data_collator,
            batch_size=self.train_batch_size,
            pin_memory=True,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator,
            batch_size=self.eval_batch_size,
            pin_memory=True,
        )

        return train_dataloader, eval_dataloader

    def test_preprocess_function(self, examples):
        """Preprocess function for test dataset"""
        batch_size = len(examples[self.text_column])
        inputs = [
            f"{self.text_column} : {x} Label : " for x in examples[self.text_column]
        ]
        model_inputs = self.tokenizer(inputs)
        targets = [str(x) for x in examples[self.label_column]]
        labels = self.tokenizer(targets)

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [self.tokenizer.pad_token_id] * (
                self.max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                self.max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][: self.max_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][: self.max_length]
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def get_test_dataloader(self):
        dataset = self.create_text_label()
        test_dataset = dataset["validation"].map(
            self.test_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=self.dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=default_data_collator,
            batch_size=self.eval_batch_size,
            pin_memory=True,
        )
        print(f"Number of Evaluating Examples: {len(test_dataset)}")
        return test_dataloader
