from itertools import groupby
from math import floor
import logging

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Subset, DataLoader

from BioDataModule import BioDataModule
from EnsembleBioDataset import EnsembleBioDataset

logger = logging.getLogger("training")


class EnsembleBioDataModule(BioDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_data(self):
        self.dataset = EnsembleBioDataset(
            self.config["hyperparams"]["num_mentions"],
            self.config["data_dir"],
            self.config["hf_model_name"],
            self.config["arch"]["add_span_tokens"],
        )
        # We wait to set the fold until prepare_data because we need the dataset
        # in order to split the data properly.
        self.num_papers = self.dataset.num_papers
        self.set_fold(self.fold)
        logger.info(f"Number of training samples: {len(self.train)}")
        logger.info(f"Number of validation samples: {len(self.val)}")
        logger.info(f"Number of test samples: {len(self.test)}")

    def collate_batch(self, batch):
        names = [
            "name",
            "label",
            "order",
        ]
        if self.config["arch"]["add_span_tokens"]:
            names.append("start_indices")
        
        collated = {x: [] for x in names}
        collated["input_ids"] = []
        collated["attention_mask"] = []
        collated["groupings"] = []
        collated["sentence_distances"] = []
        group_index = 0
        for sample in batch:
            start = group_index
            for i in range(sample['order'].shape[0]):
                collated['input_ids'].append(
                    sample['tokenizer_output'][i]['input_ids'])
                collated['attention_mask'].append(
                    sample['tokenizer_output'][i]['attention_mask'])
                group_index += 1

            for key in names:
                collated[key].append(sample[key])            
            end = group_index
            collated['groupings'].append((start, end))
            collated['sentence_distances'].extend(sample['sentence_distances'])

        collated["labels"] = torch.cat(collated["label"])
        collated["order"] = torch.cat(collated["order"])
        if self.config["arch"]["add_span_tokens"]:
            collated["start_indices"] = torch.cat(collated["start_indices"])
        collated["input_ids"] = torch.cat(collated["input_ids"])
        collated["attention_mask"] = torch.cat(collated["attention_mask"])
        collated["sentence_distances"] = torch.cat([s.unsqueeze(0) for s in collated["sentence_distances"]])
        return collated


if __name__ == "__main__":
    import json
    with open("data/jobs/lr_tuning_replication/configs/lr_tuning_replication_25_1e-4_0.conf") as f:
        config = json.load(f)

    m = EnsembleBioDataModule(config)
    m.prepare_data()

    train = m.train_dataloader()
    for b in train:
        print(b)