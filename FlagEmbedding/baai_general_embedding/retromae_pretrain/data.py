import os
import random
from copy import deepcopy
from dataclasses import dataclass

import torch.utils.data.dataset
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import DataCollatorForWholeWordMask

from .utils import tensorize_batch


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, train_data, shuffle=True):
        if isinstance(train_data, list):
            datasets = []
            for target in train_data:
                print(f"Loading {target}")
                datasets.append(self.load_dataset(target))
            self.dataset = concatenate_datasets(datasets)
        else:
            if os.path.isdir(train_data):
                datasets = []
                for target in os.listdir(train_data):
                    print(f"Loading {target}")
                    target = os.path.join(train_data, target)
                    datasets.append(self.load_dataset(target))
                self.dataset = concatenate_datasets(datasets)
            elif os.path.isfile(train_data):
                print(f"Loading {train_data}")
                self.dataset = self.load_dataset(train_data)
            else:
                self.dataset = self.load_dataset(train_data)
        if shuffle:
            print("Shuffling dataset...")
            self.dataset = self.dataset.shuffle(seed=42)

    def load_dataset(self, file):
        if file.endswith(".jsonl") or file.endswith(".json"):
            return load_dataset("json", data_files=file)["train"]
        elif os.path.isdir(file):
            return Dataset.load_from_disk(file)
        else:
            return load_dataset(file, split="train")
            # raise NotImplementedError(f"Not support this file format:{file}")

    def __getitem__(self, item):
        return self.dataset[item]["text"]

    def __len__(self):
        return len(self.dataset)


@dataclass
class RetroMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        for e in examples:

            e_trunc = self.tokenizer.encode(
                e, max_length=self.max_seq_length, truncation=True
            )
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(tokens), 128)):
                mask_set.append(self._whole_word_mask(tokens))

            text_matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = random.randint(0, min(len(tokens), 128) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(
                1 - torch.tensor(text_matrix_attention_mask)
            )

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(
            input_ids_batch, encoder_mlm_mask_batch
        )
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(
            decoder_matrix_attention_mask_batch, 0
        )

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return batch
