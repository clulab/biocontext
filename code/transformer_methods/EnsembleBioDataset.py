import os
import logging
import itertools
from copy import deepcopy
from math import log2

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from BioDataset import BioDataset

logger = logging.getLogger("training")


class EnsembleBioDataset(BioDataset):
    """
    Dataset for the biology event grounding annotations stored here:
    https://github.com/ml4ai/BioContext_corpus_private

    Sub class for modified get function. This get function creates
    pairs for up to three mentions of a specific context.
    """


    def __init__(
        self, num_mentions, *args, **kwargs
    ):
        self.num_mentions = num_mentions
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Event - Context pairs are defined through the following ordering
        # Paper | Event | Context
        # So if there are n papers, m_i events, and c_i contexts for paper i
        # then there are: sum{0, n, i}(m_i * c_i) pairings.
        # The data samples are organized as follows:
        # event_1 x context_1, ..., event_1 x context_c, event_2 x context_1, ...
        # until event_n x context_c after which we move onto the next paper
        if not index < self.last_indice[-1]:
            logger.error(
                (
                    f"Requested index: {index} is larger than max "
                    f"index: {self.last_indice[-1]}"
                )
            )
            raise IndexError

        # Last indice is also the first index of the next paper
        paper_index = 0
        while self.last_indice[paper_index] <= index:
            paper_index += 1
        paper = self.paper_data[paper_index]

        # First event-context pair index contained within this paper
        paper_index_start = self.last_indice[paper_index - 1] if paper_index != 0 else 0

        # Adjust index s.t. it's relative to the start of this paper
        index -= paper_index_start

        # For each event there is a pair with every context
        # i.e. num_event x num_context | modulo arithmetic lets us recover indicies
        # event_idx * num_context + context_idx
        event_idx = index // len(paper["context_index"])
        context_idx = index % len(paper["context_index"])

        try:
            event = paper["events"][event_idx]
        except Exception:
            print(f"index passed in: {index}")
            print(f"final paper index: {paper_index}")
            print(f"paper_index_start: {paper_index_start}")
            raise Exception

        # Get the sentence indices of the mentions for this context.
        # Also get the ranges of sentence indices between the event 
        # and each mention.
        context_mentions, spans = event["context_pairs"][context_idx]

        # If there are more than three mentions. Take only the closest 
        # three mentions, relative to the event mention
        if len(context_mentions) > self.num_mentions:
            context_mentions = context_mentions[:self.num_mentions]
            spans = spans[:self.num_mentions]

        sentence_spans = [
            deepcopy(
                [paper["sentences"][i] for i in span]
            ) for span in spans
        ]

        def overlaps_with_pair(mention, sent_index, token_index):
            # This function makes sure we don't mask any tokens in our event and
            # context pair. The elif is ugly because the linter likes it that way.
            if sent_index in [event["sentence"], mention["sentence"]]:
                if token_index >= event["start"] and token_index < event["end"]:
                    return True
                elif (
                    token_index >= mention["start"]
                    and token_index < mention["end"]
                ):
                    return True
                else:
                    return False
            else:
                return False

        for i, span in enumerate(spans):
            # Now replace other context / event tokens with corresponding tags
            cur_mention = context_mentions[i]
            for j, sent_idx in enumerate(span):
                # If there are no mentions in this sentence skip it
                if sent_idx not in paper["sentence_content"]:
                    continue

                cur_sentence = sentence_spans[i][j]
                content = paper["sentence_content"][sent_idx]
                for idx in content[self.EVT]:
                    # Check if its the event we're evaluating
                    if idx == event_idx:
                        continue
                    other_event = paper["events"][idx]
                    for tok in range(other_event["start"], other_event["end"]):
                        if not overlaps_with_pair(cur_mention, sent_idx, tok):
                            cur_sentence[tok] = self.EVT_MASK

                for (other_context_id, idx) in content[self.CON]:
                    # Check if its the context we're evaluating
                    if context_idx == other_context_id:
                        continue
                    other_context = paper["contexts"][other_context_id][idx]
                    for tok in range(other_context["start"], other_context["end"]):
                        if not overlaps_with_pair(cur_mention, sent_idx, tok):
                            cur_sentence[tok] = self.CON_MASK

        orders = []
        start_end_pairs = []
        tokenized_outputs = []
        sentence_distances = []
        for mention, sentence_span in zip(context_mentions, sentence_spans):
            # Get the sentence indices of the event/mention pair
            sentence_distances.append(abs(event["sentence"] - mention["sentence"]))
            # Find out if context is before or after event
            if mention["sentence"] < event["sentence"]:
                orders.append([1, 0])
            else:
                orders.append([0, 1])

            # Get the indices for each sentence
            if event["sentence"] < mention["sentence"]:
                evt_sent = sentence_span[0]
                con_sent = sentence_span[-1]
                evt_idx = 0
                con_idx = -1
            else:
                evt_sent = sentence_span[-1]
                con_sent = sentence_span[0]
                evt_idx = -1
                con_idx = 0

            # Calculate indices of span starts
            evt_s, evt_e = event["start"], event["end"]
            con_s, con_e = mention["start"], mention["end"]
            if evt_sent == con_sent:
                # If evt_sent == con_sent, then sentence_span should be a
                # single sentence
                assert len(sentence_span) == 1
                evt_start_idx = evt_s
                con_start_idx = con_s

                # If configured, add start / end tokens to sentences
                if self.add_span_tokens:
                    if evt_s < con_s:
                        # assert evt_e < con_s  # No overlap allowed
                        sentence_span[0] = (
                            evt_sent[:evt_s]
                            + [self.EVT_START]
                            + evt_sent[evt_s:evt_e]
                            + [self.EVT_END]
                            + evt_sent[evt_e:con_s]
                            + [self.CON_START]
                            + evt_sent[con_s:con_e]
                            + [self.CON_END]
                            + evt_sent[con_e:]
                        )
                        con_start_idx += 2

                    else:
                        # assert con_e < evt_s  # No overlap allowed
                        sentence_span[0] = (
                            evt_sent[:con_s]
                            + [self.CON_START]
                            + evt_sent[con_s:con_e]
                            + [self.CON_END]
                            + evt_sent[con_e:evt_s]
                            + [self.EVT_START]
                            + evt_sent[evt_s:evt_e]
                            + [self.EVT_END]
                            + evt_sent[evt_e:]
                        )
                        evt_start_idx += 2

            else:
                # Break the sentence apart and insert the span markers
                if self.add_span_tokens:
                    sentence_span[evt_idx] = (
                        evt_sent[:evt_s]
                        + [self.EVT_START]
                        + evt_sent[evt_s:evt_e]
                        + [self.EVT_END]
                        + evt_sent[evt_e:]
                    )
                    sentence_span[con_idx] = (
                        con_sent[:con_s]
                        + [self.CON_START]
                        + con_sent[con_s:con_e]
                        + [self.CON_END]
                        + con_sent[con_e:]
                    )
                evt_start_idx = evt_s + sum([len(x) for x in sentence_span[:evt_idx]])
                con_start_idx = con_s + sum([len(x) for x in sentence_span[:con_idx]])

            flat_sentence_span = list(itertools.chain.from_iterable(sentence_span))
            tokenized = self.tokenizer(
                flat_sentence_span,
                is_split_into_words=True,
                return_tensors="pt",
                padding="max_length",
                max_length=self.TRANSFORMER_INPUT_WIDTH,
            )

            # Perform conversion from pre-tokenized index to tokenized index
            new_con, new_evt = None, None
            token2word = tokenized.word_ids()
            for i, word_id in enumerate(token2word):
                if word_id == evt_start_idx and new_evt == None:
                    new_evt = i - 1
                if word_id == con_start_idx and new_con == None:
                    new_con = i - 1
            try:
                assert new_con != None
                assert new_evt != None
            except:
                print(flat_sentence_span)
                print(f"Original event index: {evt_s} sentence: {event['sentence']}")
                print(f"  unadjusted: {evt_start_idx}")

                print(
                    f"Original context index: {con_s} sentence: {mention['sentence']}"
                )
                print(f"  unadjusted: {con_start_idx}")
                raise
            evt_start_idx = new_evt
            con_start_idx = new_con

            # If our input is too big adjust it to fit into the transformer
            if tokenized["input_ids"].shape[1] > self.TRANSFORMER_INPUT_WIDTH:
                full_size = tokenized["input_ids"].shape[1]
                half_width = self.TRANSFORMER_INPUT_WIDTH // 2
                quarter_width = half_width // 2

                if con_start_idx < evt_start_idx:
                    first = con_start_idx
                    second = evt_start_idx
                else:
                    first = evt_start_idx
                    second = con_start_idx

                # If it's in the first window, just grab the window
                if first < half_width:
                    left = tokenized["input_ids"][0, : half_width - 1]

                # Otherwise center the window around the start span
                else:
                    start = first - quarter_width
                    end = first + quarter_width - 1  # -1 for <SEP> token
                    left = tokenized["input_ids"][0, start:end]
                    first -= start

                # If it's in the last window, just grab the window
                if second > full_size - half_width:
                    right = tokenized["input_ids"][0, -half_width:]
                    second -= full_size - (half_width * 2)

                # Otherwise center the window around the start span
                else:
                    start = second - quarter_width
                    end = second + quarter_width
                    right = tokenized["input_ids"][0, start:end]
                    second = half_width + quarter_width

                # Update start indices for new span
                if con_start_idx < evt_start_idx:
                    con_start_idx = first
                    evt_start_idx = second
                else:
                    evt_start_idx = first
                    con_start_idx = second

                tokenized["input_ids"] = torch.cat(
                    [
                        left,
                        torch.tensor([2]),  # <SEP> token
                        right,
                    ],
                    dim=0,
                )
                tokenized["input_ids"] = tokenized["input_ids"].unsqueeze(0)
                tokenized["attention_mask"] = tokenized["attention_mask"][
                    :, : self.TRANSFORMER_INPUT_WIDTH
                ]

            if self.add_span_tokens:
                start_end_pairs.append([con_start_idx, evt_start_idx])
            tokenized_outputs.append(tokenized)

        # Sample is true if context id is in our event's list of contexts
        label = [1] if context_idx in event["context_ids"] else [0]

        evt_id = event["event_id"]
        ctx_id = context_mentions[0]["context_name"]
        data_id = (paper["name"], evt_id, ctx_id)

        pair = {
            "data_id": data_id,
            "name": paper["name"],
            "label": torch.tensor(label),
            "order": torch.tensor(orders),
            "start_indices": torch.tensor(start_end_pairs)
            if self.add_span_tokens
            else [],
            "tokenizer_output": tokenized_outputs,
            "sentence_distances": torch.tensor(sentence_distances)
        }
        return pair
