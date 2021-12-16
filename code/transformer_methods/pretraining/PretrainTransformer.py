import logging
from typing import Dict, Any, Tuple, List

import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModel

logger = logging.getLogger("training")


class PretrainTransformer(pl.LightningModule):
    def __init__(self, config, num_event_types: int):
        super().__init__()
        self.config = config
        self.hyperparams = config["hyperparams"]
        self.model_name = config["hf_model_name"]
        self.transformer = AutoModel.from_pretrained(self.model_name)
        self.num_event_types = num_event_types

        # Add six more embeddings for our <EVT> and <CON> mask tokens
        # and the <START> and <END> span markers for events and contexts
        new_size = self.transformer.config.vocab_size + 6
        self.transformer.resize_token_embeddings(new_size)

        self.context_ffn = nn.Sequential(
            nn.Linear(
                self.transformer.config.hidden_size,
                self.hyperparams["hidden_layer_width"],
            ),
            nn.ReLU(),
            nn.Linear(
                self.hyperparams["hidden_layer_width"],
                self.hyperparams["hidden_layer_width"]
            )
        )

        self.event_ffn = nn.Sequential(
            nn.Linear(
                self.transformer.config.hidden_size,
                self.hyperparams["hidden_layer_width"],
            ),
            nn.ReLU(),
            nn.Linear(
                self.hyperparams["hidden_layer_width"],
                self.num_event_types
            )
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.hyperparams["learning_rate"])
        )
        return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.config,
            {
                "train_loss": 0,
                "val_loss": 0,
            },
        )
        logger.info("== Beginning model training... ==")

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.tensor, torch.tensor]:
        # Pass the inputs through the transformer and collect the embeddings for
        # the [CLS] token at position 0 in sequence.
        # Batch x embedding size
        embedding = self.transformer(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        con_indices = batch["start_indices"].unsqueeze(-1)
        con_indices = con_indices.repeat(1, 1, embedding.shape[-1])
        evt_indices = con_indices[:, 1, :].unsqueeze(1)
        con_indices = con_indices[:, 0, :].unsqueeze(1)

        context_embedding = embedding.gather(1, con_indices)
        event_embedding = embedding.gather(1, evt_indices)

        # Maybe add this combination to a config at some point
        embedding = torch.cat([context_embedding, event_embedding], dim=1)
        embedding = torch.mean(embedding, dim=1)

        context_final_embed = self.context_ffn(embedding)
        event_final_embed = self.event_ffn(embedding)
        return (context_final_embed, event_final_embed)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.tensor:
        context_embed, event_embed = self.forward(batch)

        event_loss = nn.functional.cross_entropy(
            event_embed, batch['labels']
        )

        anchor, pos, neg = [], [], []
        for i in range(batch['context_pairs'].shape[0] // 4):
            for j in range(4):
                anchor.append(torch.clone(context_embed[i + j, :].unsqueeze(0)))
                if j < 2:
                    # 1 - 1 == 0 and 1 - 0 == 1, so we map to the other example
                    pos.append(torch.clone(context_embed[i + (j - 1), :].unsqueeze(0)))
                    # if j==0 then j + 2 == 2 | if j==1 then j + 2 == 3
                    neg.append(torch.clone(context_embed[i + (j + 2), :].unsqueeze(0)))
                else:
                    # 5 - 2 == 3 and 5 - 3 == 2, so we map to the other example
                    pos.append(torch.clone(context_embed[i + (5 - j), :].unsqueeze(0)))
                    # if j==3 then j % 2 == 1 | if j==2 then j % 2 == 0
                    neg.append(torch.clone(context_embed[i + (j % 2), :].unsqueeze(0)))
        anchor = torch.stack(anchor)
        pos = torch.stack(pos)
        neg = torch.stack(neg)
        context_loss = nn.functional.triplet_margin_loss(
            anchor, pos, neg, reduction='sum'
        )

        alpha = self.config['hyperparams']['alpha']
        loss = alpha * event_loss + (1 - alpha) * context_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        context_embed, event_embed = self.forward(batch)

        event_loss = nn.functional.cross_entropy(
            event_embed, batch['labels']
        )

        anchor, pos, neg = [], [], []
        for i in range(batch['context_pairs'].shape[0] // 4):
            for j in range(4):
                anchor.append(torch.clone(context_embed[i + j, :].unsqueeze(0)))
                if j < 2:
                    # 1 - 1 == 0 and 1 - 0 == 1, so we map to the other example
                    pos.append(torch.clone(context_embed[i + (j - 1), :].unsqueeze(0)))
                    # if j==0 then j + 2 == 2 | if j==1 then j + 2 == 3
                    neg.append(torch.clone(context_embed[i + (j + 2), :].unsqueeze(0)))
                else:
                    # 5 - 2 == 3 and 5 - 3 == 2, so we map to the other example
                    pos.append(torch.clone(context_embed[i + (5 - j), :].unsqueeze(0)))
                    # if j==3 then j % 2 == 1 | if j==2 then j % 2 == 0
                    neg.append(torch.clone(context_embed[i + (j % 2), :].unsqueeze(0)))
        anchor = torch.stack(anchor)
        pos = torch.stack(pos)
        neg = torch.stack(neg)
        context_loss = nn.functional.triplet_margin_loss(
            anchor, pos, neg, reduction='sum'
        )

        alpha = self.config['hyperparams']['alpha']
        loss = alpha * event_loss + (1 - alpha) * context_loss

        self.log("context_loss", context_loss)
        self.log("event_loss", event_loss)
        self.log("validation_loss", loss)
        # Pass logits and labels so that we can calc f1 for validation epoch
        return event_embed, batch['labels']

    def training_epoch_end(self, outputs: List[Any]) -> None:
        logger.info(f"Completed training epoch {1+self.current_epoch}")

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        y_pred = torch.cat([x[0].argmax(dim=1) for x in outputs]).cpu()
        y_true = torch.cat([x[1] for x in outputs]).cpu()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=1
        )
        self.log("validation_precision", precision)
        self.log("validation_recall", recall)
        self.log("validation_f1", f1)
        logger.info(f"Completed validation epoch, f1: {f1:.4}")

    