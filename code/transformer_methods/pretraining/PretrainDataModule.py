from math import floor
import logging

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Subset, DataLoader

from pretraining.PretrainDataset import PretrainDataset

logger = logging.getLogger("training")


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        self.dataset = PretrainDataset(
            self.config["data_dir"],
            self.config["hf_model_name"]
        )
        self.num_papers = self.dataset.num_papers
        self.num_event_types = self.dataset.num_event_types

        papers = np.random.permutation(np.arange(0, self.num_papers))
        test_frac = round(self.config['run_config']['test_percentage'], 4)
        val_frac = round(self.config['run_config']['validation_percentage'], 4)
        train_frac = 1 - test_frac - val_frac
        
        train_papers = papers[:floor(len(papers)*train_frac)]
        non_train_papers = papers[floor(len(papers)*train_frac):]
        val_ratio = val_frac / (val_frac + test_frac)
        val_papers = non_train_papers[:floor(len(non_train_papers)*val_ratio)]
        test_papers = non_train_papers[floor(len(non_train_papers)*val_ratio):]

        # Iterate over all samples and place them into their respective splits
        val_ranges = [self.dataset.get_paper_range(paper) for paper in val_papers]
        test_ranges = [self.dataset.get_paper_range(paper) for paper in test_papers]
        train_ranges = [self.dataset.get_paper_range(paper) for paper in train_papers]

        train, val, test = [], [], []
        for start, end in val_ranges:
            for i in range(start, end):
                val.append(i)
        for start, end in test_ranges:
            for i in range(start, end):
                test.append(i)
        for start, end in train_ranges:
            for i in range(start, end):
                train.append(i)

        self.train = np.array(train)
        self.val = np.array(val)
        self.test = np.array(test)
        
        logger.info(f"Number of training samples: {len(self.train)}")
        logger.info(f"Number of validation samples: {len(self.val)}")
        logger.info(f"Number of test samples: {len(self.test)}")

    def collate_batch(self, batch):
        names = [
            "name",
            "labels",
            "order",
            "start_indices",
        ]

        context_id = 0
        collated = {x: [] for x in names}
        collated['input_ids'] = []
        collated['attention_mask'] = []
        collated['context_pairs'] = []
        for sample in batch:
            for key in names:
                collated[key].append(sample[key])

            # Unpack output from tokenizer
            for i in range(sample['order'].shape[0]):
                collated['input_ids'].append(
                    sample['tokenizer_output'][i]['input_ids'])
                collated['attention_mask'].append(
                    sample['tokenizer_output'][i]['attention_mask'])

            # We know context mentions come in pairs so add two before
            # updating the context id then repeat
            collated["context_pairs"].append([context_id])
            collated["context_pairs"].append([context_id])
            context_id += 1
            collated["context_pairs"].append([context_id])
            collated["context_pairs"].append([context_id])
            context_id += 1
            
        collated["labels"] = torch.cat(collated["labels"])
        collated["order"] = torch.cat(collated["order"])
        collated["start_indices"] = torch.cat(collated["start_indices"])
        collated["input_ids"] = torch.cat(collated["input_ids"])
        collated["attention_mask"] = torch.cat(collated["attention_mask"])
        collated["context_pairs"] = torch.cat(
            [torch.tensor(x) for x in collated["context_pairs"]]
        )
        return collated

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.dataset, self.train),
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["train_thread_count"],
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.dataset, self.val),
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["eval_thread_count"],
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_batch,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.dataset, self.test),
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["eval_thread_count"],
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_batch,
        )
