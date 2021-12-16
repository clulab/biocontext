from math import floor
import logging

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Subset, DataLoader

from BioDataset import BioDataset

logger = logging.getLogger("training")


class BioDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fold = config["run_config"]["test_paper_index"]
        self.num_val_papers = config["run_config"]["validation_paper_count"]

    def set_fold(self, new_fold: int):
        self.fold = new_fold

        indices = np.arange(len(self.dataset))
        # Either a validation run or a test run
        if self.config["run_config"]["test_type"] == "validation":
            assert self.fold >= 0 and self.fold < 10
            val_papers = [
                x % 10  # Only wrap around the first 10 papers [0, 9]
                for x in range(self.fold, self.fold + self.num_val_papers)
            ]
            logger.info(
                (
                    "== Setting validation papers to "
                    f"{', '.join([self.dataset.paper_data[x]['name'] for x in val_papers])} =="
                )
            )

            val_ranges = [self.dataset.get_paper_range(paper) for paper in val_papers]
            train, val = [], []
            for sample in indices:
                flag = False
                for start, end in val_ranges:
                    if sample >= start and sample < end:
                        flag = True
                if not flag:
                    train.append(sample)
                else:
                    val.append(sample)
            self.train = np.array(train)
            self.val = np.array(val)
            self.test = np.array([])

        elif self.config["run_config"]["test_type"] == "test":
            test_index = self.fold
            assert test_index >= 10 and test_index < 27
            logger.info(
                (
                    "== Setting held out test paper to "
                    f"{self.dataset.paper_data[test_index]['name']} =="
                )
            )

            test_range = self.dataset.get_paper_range(self.fold)
            # Get all values inside held out test paper
            self.test = np.where(
                (indices >= test_range[0]) & (indices < test_range[1])
            )[0]
            # Get all values outside held out test paper
            rest = np.where((indices >= test_range[1]) | (indices < test_range[0]))[0]
            self.train = np.array(rest)
            # Use single training sample for validation set.
            # pytorch lightning gets upset if validation is empty.
            self.val = np.array([rest[0]])

        else:
            raise ValueError(
                "test_type parameter must be either 'validation' or 'test'"
            )

    def prepare_data(self):
        self.dataset = BioDataset(
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
            "input_ids",
            "attention_mask",
        ]
        if self.config["arch"]["add_span_tokens"]:
            names.append("start_indices")
        collated = {x: [] for x in names}
        for sample in batch:
            for key in names:
                collated[key].append(sample[key])
        collated["labels"] = torch.cat(collated["label"])
        collated["order"] = torch.cat(collated["order"])
        if self.config["arch"]["add_span_tokens"]:
            collated["start_indices"] = torch.cat(collated["start_indices"])
        collated["input_ids"] = torch.cat(collated["input_ids"])
        collated["attention_mask"] = torch.cat(collated["attention_mask"])
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
