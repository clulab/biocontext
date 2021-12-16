from argparse import ArgumentError
import logging
from typing import List, Any

import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger("training")


class SimpleTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hyperparams = config["hyperparams"]
        self.model_name = config["hf_model_name"]
        self.transformer = AutoModel.from_pretrained(self.model_name)

        # Add six more embeddings for our <EVT> and <CON> mask tokens
        # and the <START> and <END> span markers for events and contexts
        new_size = self.transformer.config.vocab_size + 6
        self.transformer.resize_token_embeddings(new_size)

        if config["arch"]["order_linear_proj_enabled"]:
            # Triple the size of the CLS embedding, for the
            # Main embedding, Context first embedding, Context last embedding
            self.projection = nn.Linear(
                self.transformer.config.hidden_size * 3,
                self.hyperparams["hidden_layer_width"],
            )
        else:
            self.projection = nn.Linear(
                self.transformer.config.hidden_size,
                self.hyperparams["hidden_layer_width"],
            )

        if config["arch"]["ensemble"]["enabled"]:
            if config["arch"]["ensemble"]["type"] == "self-attention":
                # The pseudo-attention aggregation is similar to the way multi head attention is reduced to the original head dimentionality
                # using a linear projection.
                # We will "aggregate" at most 10 mentions down to the size of one
                self.pseudo_attention = nn.Sequential(
                    nn.Linear(self.transformer.config.hidden_size * 10, self.transformer.config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.transformer.config.hidden_size, self.transformer.config.hidden_size),
                    nn.Dropout(),
                )
        self.final_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hyperparams["hidden_layer_width"], 2),
        )

    def forward(self, batch):
        # Pass the inputs through the transformer and collect the embeddings for
        # the [CLS] token at position 0 in sequence.
        # Batch x embedding size
        embedding = self.transformer(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state

        if self.config["arch"]["use_span_not_cls"]:
            con_indices = batch["start_indices"].unsqueeze(-1)
            con_indices = con_indices.repeat(1, 1, embedding.shape[-1])
            evt_indices = con_indices[:, 1, :].unsqueeze(1)
            con_indices = con_indices[:, 0, :].unsqueeze(1)

            context_embedding = embedding.gather(1, con_indices)
            event_embedding = embedding.gather(1, evt_indices)

            # Maybe add this combination to a config at some point
            embedding = torch.cat([context_embedding, event_embedding], dim=1)
            embedding = torch.mean(embedding, dim=1)
        else:
            embedding = embedding[:, 0, :]

        if self.config["arch"]["order_linear_proj_enabled"]:
            embed_size = embedding.shape[-1]
            embedding = embedding.repeat(1, 3)
            # Expand last dim to maintain 2d status for broadcasting
            mask = batch["order"].unsqueeze(-1)

            # Mask out the section of embedding that doesn't match our ordering
            # ie. if context comes first we want yyy | yyy | 000 (mask = 1, 0)
            #     if event comes first we want   yyy | 000 | yyy (mask = 0, 1)
            embedding[:, embed_size : embed_size * 2] *= mask[:, 0]
            embedding[:, embed_size * 2 :] *= mask[:, 1]

        if (self.config["arch"]["ensemble"]["enabled"]):
            final = []

            # Un-weighted average
            if self.config["arch"]["ensemble"]["type"] == "average":
                for group in batch["groupings"]:
                    final.append(
                        torch.mean(embedding[group[0] : group[1], :], dim=0).unsqueeze(0)
                    )
            # Weighted average from the inverse of sentence distance
            elif self.config["arch"]["ensemble"]["type"] == "distance":
                for group in batch["groupings"]:
                    # Fetch the sentence distances
                    sent_distances = batch["sentence_distances"][group[0] : group[1]]
                    # Compute the inverse of the distances and normalize
                    inverse_distances = 1 / (sent_distances + 1e-8)
                    weights = (inverse_distances / inverse_distances.sum()).unsqueeze(0)
                    # Weighted average
                    w_avg = (weights @ embedding[group[0] : group[1], :])
                    final.append(w_avg)
            elif self.config["arch"]["ensemble"]["type"] == "self-attention":
                for group in batch["groupings"]:
                    # Number of tensors in the group
                    num_tensors = group[1] - group[0]
                    to_pad = 10 - num_tensors

                    elems = embedding[group[0] : group[1], :]

                    pad = torch.zeros(to_pad, elems.shape[-1], device=self.device)

                    x = torch.cat([elems, pad], dim=0).view(1, -1)

                    # Pass the padded embedding through the pseudo-attention layer
                    w_attn = self.pseudo_attention(x)

                    # # Query, key, value
                    # Q = self.WQ(embedding)
                    # # K = self.WK(embedding)
                    # K = self.K
                    # V = self.WV(embedding)

                    # import torch.nn.functional as F
                    # import math

                    # # Compute the attention weights
                    # alphas = F.softmax(K(Q)/ math.sqrt(self.transformer.config.hidden_size), dim=0)
                    # # Compute the aggregated vector
                    # w_attn = alphas.T @ V

                    final.append(w_attn)

            else:
                # If other type of "ensemble" is requested, do a fwd pass over the whole batch
                # The aggregation will be done after the logits are computed
                final.append(embedding)

            embedding = torch.cat(final, dim=0)

        after_ffn = self.projection(embedding)
        return self.final_ffn(after_ffn)

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
                "validation_f1": 0,
                "validation_precision": 0,
                "validation_recall": 0,
            },
        )
        logger.info("== Beginning model training... ==")

    def __calculate_batch_loss(self, batch):
        logits = self.forward(batch)
        weights = torch.tensor(
            [self.hyperparams["negative_sample_weight"], 1.0], device=self.device
        )
        if self.config["arch"]["ensemble"]["enabled"] and \
           self.config["arch"]["ensemble"]["type"] in {
            "vote", "post_distance", "one_hit", "logits"
           }:
            
            ensemble_type = self.config["arch"]["ensemble"]["type"]
            expanded_labels = torch.zeros(
                logits.shape[0], device=self.device, dtype=batch["labels"].dtype
            )
            for i, group in enumerate(batch["groupings"]):
                expanded_labels[group[0] : group[1]] = batch["labels"][i]

            loss = nn.functional.cross_entropy(
                logits, expanded_labels, weight=weights, reduction="none"
            )
            # If using voting-based ensemble. Each sample must have equal
            # weight in the final loss term. This means we need to divide
            # the loss values by the number of pairs for each sample
            for i, group in enumerate(batch["groupings"]):

                if ensemble_type in {"vote", "one_hit"}:
                    g_weights = 1 / (group[1] - group[0])
                elif ensemble_type == "post_distance":
                    # Fetch the sentence distances
                    sent_distances = batch["sentence_distances"][group[0] : group[1]]
                    # Compute the inverse of the distances and normalize
                    inverse_distances = 1 / (sent_distances + 1e-8)
                    g_weights = (inverse_distances / inverse_distances.sum())
                elif ensemble_type in {"logits"}:
                    max_vals = logits[group[0] : group[1]].max(dim=1)[0]
                    g_weights = max_vals / max_vals.sum()

                loss[group[0] : group[1]] *= g_weights

            loss = torch.mean(loss, dim=0)
            
        # This is the normal case for no aggregation whatsoever. The dataloader gives back a representative
        # Which is the span of text with the closest sentence distance
        else:
            loss = nn.functional.cross_entropy(logits, batch["labels"], weight=weights)

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.__calculate_batch_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, logits = self.__calculate_batch_loss(batch)

        self.log("val_loss", loss)
        if self.config["arch"]["ensemble"]["enabled"]:
            return logits, batch["labels"], batch["data_id"],  batch["groupings"], batch["sentence_distances"]
        else:
            return logits, batch["labels"], batch["data_id"]

    def test_step(self, batch, batch_idx):

        loss, logits = self.__calculate_batch_loss(batch)

        self.log("test_loss", loss)
        if self.config["arch"]["ensemble"]["enabled"]:
            return logits, batch["labels"], batch["data_id"],  batch["groupings"], batch["sentence_distances"]
        else:
            return logits, batch["labels"], batch["data_id"]
            

    def training_epoch_end(self, outputs: List[Any]) -> None:
        logger.info(f"Completed training epoch {1+self.current_epoch}")

    def __calculate_predictions(self, outputs, average="macro"):
        y_true = torch.cat([x[1] for x in outputs]).cpu()
        # If we are doing ensemble voting, convert logits to be voted
        # decision over all pairs for a context-event sample
        if self.config["arch"]["ensemble"]["enabled"] and \
        self.config["arch"]["ensemble"]["type"] in {
            "vote",
            "post_distance",
            "one_hit",
            "logits",
        }:

            ensemble_type = self.config["arch"]["ensemble"]["type"]
    
            voted_labels = []
            for out in outputs:
                logits = out[0]
                groups = out[3]
                distances = out[4]
                for group in groups:
                    votes = logits[group[0] : group[1], :].argmax(dim=1)
                    if ensemble_type == "vote":
                        voted_labels.append(votes.mode()[0])  # Get most voted label
                    elif ensemble_type == "one_hit":
                        voted_labels.append((votes == 1).any().int())
                    elif ensemble_type == "post_distance":
                        inverse_sent_distances = 1 / (distances[group[0] : group[1]] + 1e-8)
                        weights = (inverse_sent_distances / inverse_sent_distances.sum())
                        vote_mass = torch.tensor([0., 0.], device=self.device)
                        for vote, weight in zip(votes, weights):
                            vote_mass[vote] += weight
                        voted_labels.append(vote_mass.argmax())
                    elif ensemble_type in {"logits"}:
                        local_logits = logits[group[0] : group[1], :]
                        vote_mass = torch.tensor([0., 0.], device=self.device)
                        for vote, ls in zip(votes, local_logits):
                            weight = ls[vote]
                            vote_mass[vote] += weight
                        voted_labels.append(vote_mass.argmax())
                    
                        
            y_pred = torch.stack(voted_labels).cpu()


        else:
            y_pred = torch.cat([x[0] for x in outputs]).argmax(dim=1).cpu()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=1
        )

        return precision, recall, f1, support, y_true, y_pred

    def validation_epoch_end(self, outputs: List[Any]) -> None:

        precision, recall, f1, _, y_true, y_pred = self.__calculate_predictions(outputs, average=None)
       
        self.log("validation_precision", precision[1])
        self.log("validation_recall", recall[1])
        self.log("validation_f1", f1[1])
        logger.info(f"Completed validation epoch, f1: {f1[1]:.4}")

    def test_epoch_end(self, outputs: List[Any]) -> None:

        corrected_data_ids = list()
        for output in outputs:
            data_ids = output[2]
            corrected_data_ids.extend('\t'.join(di) for di in data_ids)

        

        precision, recall, f1, support, y_true, y_pred = self.__calculate_predictions(outputs, average=None)

        assert len(corrected_data_ids) == len(y_true)

        tensorboard = self.logger.experiment
        tensorboard.add_text("test_labels", '\n'.join(y_true.numpy().astype(str)))
        tensorboard.add_text("test_predictions", '\n'.join(y_pred.numpy().astype(str)))
        tensorboard.add_text("test_data_ids", '\n'.join(corrected_data_ids))
       
        self.log("test_precision_positive", precision[1])
        self.log("test_recall_positive", recall[1])
        self.log("test_f1_positive", f1[1])
        self.log("test_support_positive", support[1])
        self.log("test_precision_negative", precision[0])
        self.log("test_recall_negative", recall[0])
        self.log("test_f1_negative", f1[0])
        self.log("test_support_negative", support[0])
       
