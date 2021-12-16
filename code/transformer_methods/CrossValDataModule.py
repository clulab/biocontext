from BioDataset import BioDataset
import pytorch_lightning as pl
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger("crossval_data_module")  

class CrossValDataModule(pl.LightningDataModule):
    """
    This class is used to load the data for the cross validation.
    """

    def __init__(self, config, fold_idx, k = 5, all_data=False):
        super().__init__()
        self.config = config
        self.fold_idx = fold_idx
        self.k = k
        self.all_data = all_data

    def prepare_data(self):
        # Instantiate the dataset from the configuration
        self.dataset = BioDataset(
            self.config["data_dir"],
            self.config["hf_model_name"],
            self.config["arch"]["add_span_tokens"],
        )
        # We wait to set the fold until prepare_data because we need the dataset
        # in order to split the data properly.
        self.num_papers = self.dataset.num_papers
        self.set_fold(self.fold_idx, self.all_data)
        logger.info(f"Number of training samples: {len(self.train)}")
        logger.info(f"Number of validation samples: {len(self.val)}")
        logger.info(f"Number of test samples: {len(self.test)}")

    def get_fold_papers(self, fold, is_validation = False):
        fold_size = self.num_papers // (self.k+2) # We do k+2 fold cross validation

        # The first two folds are for validation
        val_indices = list(range(fold_size * 2))
        test_indices = list(range((fold+2)*fold_size, (fold+3)*fold_size))
        # The remaining are for CV, minus the "fold" index are for training
        train_indices = [i for i in range(fold_size * 2, self.num_papers) if i not in test_indices]

        train_names = [self.dataset.paper_data[x]['name'] for x in train_indices]
        val_names = [self.dataset.paper_data[x]['name'] for x in val_indices]
        test_names = [self.dataset.paper_data[x]['name'] for x in test_indices]

        if is_validation:
            train_names += test_names
            train_indices += test_indices

        return train_indices, train_names, val_indices, val_names, test_indices, test_names

    def set_fold(self, fold: int, is_validation: bool):

        indices = np.arange(len(self.dataset))

        

        train_papers, train_names, val_papers, val_names, test_papers, test_names = self.get_fold_papers(fold, is_validation)

        

        # Logginng statements
        logger.info(f"== Doing a validation pass ==")
        logger.info(
               f"== Setting validation papers to {', '.join(val_names)} =="
        )

        logger.info(
                f"== Setting training papers to {', '.join(train_names)} =="
        )

        logger.info(
                f"== Setting test paper to {', '.join(test_names)} =="
        )
        ######################

        # We split the data into train, validation and test
        val_ranges = [self.dataset.get_paper_range(paper) for paper in val_papers]
        test_range = [self.dataset.get_paper_range(paper) for paper in test_papers]

        train, val, test = [], [], []

        for sample in indices:
            is_val, is_test = False, False
            for start, end in val_ranges:
                if sample >= start and sample < end:
                    is_val = True
                    break
            if is_val:
                val.append(sample)
            else:
                for start, end in test_range:
                    if sample >= start and sample < end:
                        is_test = True
                        break
                if is_test:
                    test.append(sample)
                else: # Then, this is a training instance
                    train.append(sample)
           
        # Set the appropriate indices to the folds
        if not is_validation:
            self.train = np.array(train + val) # Remember to add the validation set to the training set, as discussed by the group
        else:
            self.train = np.array(train)
        self.val = np.array(val)
        self.val = np.array(val)
        self.test = np.array(test)

    def collate_batch(self, batch):
        names = [
            "data_id",
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


# if __name__ == "__main__":
#     import json
#     with open("data/jobs/lr_tuning_replication/configs/lr_tuning_replication_25_1e-4_0.conf") as f:
#         config = json.load(f)

#     m = CrossValDataModule(config, 4)
#     m.prepare_data()

#     train = m.train_dataloader()
#     for b in train:
#         print(b)