import os
import logging
import itertools
from copy import deepcopy
from math import log2

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger("training")


class BioDataset(Dataset):
    """
    Dataset for the biology event grounding annotations stored here:
    https://github.com/ml4ai/BioContext_corpus_private
    """

    # Size of transformer's attention masks
    TRANSFORMER_INPUT_WIDTH = 512

    # Named indexes into sentence index for readability
    CON = 0  # Context
    EVT = 1  # Event

    # Masks for non-pair context mentions and event mentions
    CON_MASK = "<CONTEXT>"
    EVT_MASK = "<EVENT>"

    # Tokens for event start/end
    EVT_START = "<EVT_START>"
    EVT_END = "<EVT_END>"
    CON_START = "<CON_START>"
    CON_END = "<CON_END>"

    def __init__(
        self,
        data_dir: str,
        tokenizer_name: str,
        add_span_tokens: bool,
        closest_sentence: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, add_prefix_space=True
        )
        new_mask_tokens = {
            "additional_special_tokens": [
                self.CON_MASK,
                self.EVT_MASK,
                self.EVT_START,
                self.EVT_END,
                self.CON_START,
                self.CON_END,
            ]
        }
        self.data_dir = data_dir
        self.tokenizer.add_special_tokens(new_mask_tokens)
        self.add_span_tokens = add_span_tokens
        self.closest_sentence = closest_sentence

        if not os.path.isdir(data_dir):
            logger.error(f"ERROR: Provided data_dir path is not a directory.")
            raise FileExistsError

        paper_paths = []
        for folder in os.listdir(data_dir):
            if folder[:3] == "PMC":
                paper_paths.append(folder)


        if len(paper_paths) == 0:
            logger.error(
                f"ERROR: No paper dirs starting with 'PMC' found at path: {data_dir}"
            )
            raise FileNotFoundError

        if len(paper_paths) < 4:
            logger.warning(f"Only found {len(paper_paths)} training may not work")

        self.paper_data = []
        self.number_of_pairs = 0
        for paper in paper_paths:
            logger.info(f"Parsing information about paper: {paper}...")

            sentence_path = os.path.join(data_dir, paper, "sentences.txt")
            if not os.path.exists(sentence_path):
                logger.warning(
                    f"No sentence file found for paper dir: {paper} skipping..."
                )
                continue

            mention_path = os.path.join(data_dir, paper, "mention_intervals.txt")
            if not os.path.exists(mention_path):
                logger.warning(
                    f"No context mention file found for paper dir: {paper} skipping..."
                )
                continue

            manual_mention_path = os.path.join(
                data_dir, paper, "manual_context_mentions.tsv"
            )
            if not os.path.exists(mention_path):
                logger.warning(
                    f"No manual context mention file found for paper dir: {paper} skipping..."
                )
                continue

            annotation_path = os.path.join(
                data_dir, paper, "annotated_event_intervals.tsv"
            )
            if not os.path.exists(annotation_path):
                logger.warning(
                    f"No annotation file found for paper dir: {paper} skipping..."
                )
                continue

            raw_sentences = []
            with open(sentence_path, "r") as in_file:
                # Sentences should already be pre-tokenized
                for line in in_file:
                    cur = line.strip().split()
                    raw_sentences.append(cur)

            sentence_content = {}  # from sentence index to contexts and events
            context_index = {}  # from context name to context index
            contexts = {}  # from context index to list of occurences
            with open(mention_path, "r") as in_file:
                # Each line should be:
                # sentence_index start_token%end_token%span%identifier (repeat...)
                #   or
                # sentence_index
                for line in in_file:
                    cur = line.strip().split()
                    if len(cur) == 1:
                        continue

                    sent_idx = int(cur[0])
                    for context in cur[1:]:
                        temp = context.split("%")
                        sent = raw_sentences[sent_idx]
                        start, end = int(temp[0]), int(temp[1])
                        end += 1  # add one to convert from [x, y] to [x, y)
                        assert start >= 0 and start < len(sent)
                        assert end >= 0 and end < len(sent)

                        new_mention = {}
                        new_mention["sentence"] = sent_idx  # int
                        new_mention["start"] = start  # int
                        new_mention["end"] = end  # int
                        new_mention["context_name"] = temp[3]  # str

                        # Update reverse index before adding mention to list
                        if temp[3] in context_index:
                            context_id = context_index[temp[3]]
                            contexts[context_id].append(new_mention)
                        else:
                            context_index[temp[3]] = len(context_index)
                            context_id = context_index[temp[3]]
                            contexts[context_id] = [new_mention]

                        # add context mention to reverse index
                        #  format is tuple (context id, index in contexts)
                        if sent_idx not in sentence_content:
                            sentence_content[sent_idx] = ([], [])
                        sentence_content[sent_idx][self.CON].append(
                            (context_id, len(contexts[context_id]) - 1)
                        )

            with open(manual_mention_path, "r") as in_file:
                # each line should be:
                # sentence_index \t start_token-end_token \t identifier
                for line in in_file:
                    cur = line.strip().split("\t")
                    sent_idx = int(cur[0])
                    sent = raw_sentences[sent_idx]
                    start, end = [int(_) for _ in cur[1].split("-")]
                    end += 1  # add one to convert from [x, y] to [x, y)
                    assert start >= 0 and start < len(sent)
                    assert end >= 0 and end < len(sent)

                    new_mention = {}
                    new_mention["sentence"] = sent_idx  # int
                    new_mention["start"] = start  # int
                    new_mention["end"] = end  # int
                    new_mention["context_name"] = cur[2]  # str

                    # Update reverse index before adding mention to list
                    if cur[2] in context_index:
                        context_id = context_index[cur[2]]
                        contexts[context_id].append(new_mention)
                    else:
                        context_index[cur[2]] = len(context_index)
                        context_id = context_index[cur[2]]
                        contexts[context_id] = [new_mention]

                    # add context mention to reverse index
                    #  format is tuple (context id, index in contexts)
                    if sent_idx not in sentence_content:
                        sentence_content[sent_idx] = ([], [])
                    sentence_content[sent_idx][self.CON].append(
                        (context_id, len(contexts[context_id]) - 1)
                    )

            events = []
            positive_pairs = list()
            event_ids = list()
            with open(annotation_path, "r") as in_file:
                # Each line should be:
                # sentence_index \t start_token-end_token \t identifier(,identifier,...)
                for line in in_file:
                    cur = line.strip().split("\t")
                    sent_idx = int(cur[0])
                    start, end = [int(_) for _ in cur[1].split("-")]
                    end += 1  # add one to convert from [x, y] to [x, y)
                    assert start >= 0 and start < len(raw_sentences[sent_idx])
                    assert end >= 0 and end < len(raw_sentences[sent_idx])

                    evt_id = f"E{sent_idx}_{start}_{end}"
                    event_ids.append(evt_id)

                    new_event = {}
                    new_event["sentence"] = sent_idx  # int
                    new_event["start"] = start  # int
                    new_event["end"] = end  # int
                    new_event["event_id"] = evt_id  # str
                    if len(cur) > 2:
                        local_contexts = cur[2].split(",")
                        new_event["context_ids"] = [
                            context_index[x] for x in local_contexts
                        ]  # List[int]
                        for ctx_id in local_contexts:
                            positive_pairs.append((paper, evt_id, ctx_id))
                    else:
                        new_event["context_ids"] = []

                    # For each context, sort the sentence spans by distance to the event
                    spans = dict()
                    for context_id in contexts:
                        context_mentions = sorted(
                            contexts[context_id],
                            key=lambda x: (new_event["sentence"] - x["sentence"]) ** 2,
                        )
                        span_list = []
                        for context in context_mentions:
                            # Get all sentences between context and event
                            if context["sentence"] > new_event["sentence"]:
                                # Have to add one because range is [x, y)
                                span_list.append(
                                    range(
                                        new_event["sentence"], context["sentence"] + 1
                                    )
                                )
                            else:  # This case captures when they are equal
                                span_list.append(
                                    range(
                                        context["sentence"], new_event["sentence"] + 1
                                    )
                                )
                        # Save span in event for later use
                        spans[context_id] = (context_mentions, span_list)
                    new_event["context_pairs"] = spans

                    events.append(new_event)

                    # add event mention to reverse index
                    #  format is index into 'events' list
                    if sent_idx not in sentence_content:
                        sentence_content[sent_idx] = ([], [])
                    sentence_content[sent_idx][self.EVT].append(len(events) - 1)

            negative_pairs = list()
            for evt_id in event_ids:
                for ctxs in contexts.values():
                    for ctx in ctxs:
                        ctx_id = ctx['context_name']
                        if (paper, evt_id, ctx_id) not in positive_pairs:
                            negative_pairs.append((paper, evt_id, ctx_id))

            # We do binary classification between all event-context pairs
            self.number_of_pairs += len(events) * len(context_index)
            cur_paper = {
                "name": paper,  # str
                "sentences": raw_sentences,  # List[List[str]]
                "events": events,  # List[Dict[str, obj]]
                "context_index": context_index,  # Dict[str, int]
                "contexts": contexts,  # Dict[int, List[obj]]
                "sentence_content": sentence_content,  # Dict[Tuple(List[int], List[int, int])]
                "id_pairs": {"positive":positive_pairs, "negative":negative_pairs},  # Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]
            }

            self.paper_data.append(cur_paper)
        self.num_papers = len(self.paper_data)
        self.num_contexts = len(context_index)

        # Sort papers for consistent indexing. We want the original indexing to
        # be consistent so that we can control the randomizing of our indexing
        # later using random seeds.
        self.paper_data.sort(key=lambda x: int(x["name"][3:]))
        self.paper_pairs = {d['name']:d['id_pairs'] for d in self.paper_data}

        # Generate folds (or per paper indices)
        # i.e. each paper covers a section of the total number of indices
        #  we want a map from paper -> subsection of indices so that we
        #  can easily perform leave one out CV at the paper level
        self.last_indice = []
        prev_end = 0
        for paper in self.paper_data:
            prev_end += len(paper["events"]) * len(paper["context_index"])
            self.last_indice.append(prev_end)

    def get_paper_range(self, paper_index):
        # Get range of indices for a single paper
        assert paper_index >= 0 and paper_index < self.num_papers
        if paper_index != 0:
            start = self.last_indice[paper_index - 1]
        else:
            start = 0
        end = self.last_indice[paper_index]
        return (start, end)

    def __len__(self):
        return self.number_of_pairs

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

        # Get the sentence index of the closest mention to the event sentence
        # for this context. Also get the range of sentence indices between the
        # event and the closest context mention.
        closest_context, span = event["context_pairs"][context_idx]
        closest_context = closest_context[0]  # grab the closest to the event
        span = span[0]  # grab the matching span
        sentence_span = deepcopy([paper["sentences"][i] for i in span])

        def overlaps_with_pair(sent_index, token_index):
            # This function makes sure we don't mask any tokens in our event and
            # context pair. The elif is ugly because the linter likes it that way.
            if sent_index in [event["sentence"], closest_context["sentence"]]:
                if token_index >= event["start"] and token_index < event["end"]:
                    return True
                elif (
                    token_index >= closest_context["start"]
                    and token_index < closest_context["end"]
                ):
                    return True
                else:
                    return False
            else:
                return False

        # Now replace other context / event tokens with corresponding tags
        for i, sent_idx in enumerate(span):
            # If there are no mentions in this sentence skip it
            if sent_idx not in paper["sentence_content"]:
                continue

            cur_sentence = sentence_span[i]
            content = paper["sentence_content"][sent_idx]
            for idx in content[self.EVT]:
                # Check if its the event we're evaluating
                if idx == event_idx:
                    continue
                other_event = paper["events"][idx]
                for tok in range(other_event["start"], other_event["end"]):
                    if not overlaps_with_pair(sent_idx, tok):
                        cur_sentence[tok] = self.EVT_MASK

            for (other_context_id, idx) in content[self.CON]:
                # Check if its the context we're evaluating
                if context_idx == other_context_id:
                    continue
                other_context = paper["contexts"][other_context_id][idx]
                for tok in range(other_context["start"], other_context["end"]):
                    if not overlaps_with_pair(sent_idx, tok):
                        cur_sentence[tok] = self.CON_MASK

        # Find out if context is before or after event
        if closest_context["sentence"] < event["sentence"]:
            order = [1, 0]  # Context before event
        else:
            order = [0, 1]  # Event before context

        # Get the indices for each sentence
        if event["sentence"] < closest_context["sentence"]:
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
        con_s, con_e = closest_context["start"], closest_context["end"]
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
                f"Original context index: {con_s} sentence: {closest_context['sentence']}"
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

        # Sample is true if context id is in our event's list of contexts
        label = [1] if context_idx in event["context_ids"] else [0]

        evt_id = event["event_id"]
        ctx_id = closest_context["context_name"]
        data_id = (paper["name"], evt_id, ctx_id)

        pair = {
            "data_id": data_id,
            "name": paper["name"],
            "label": torch.tensor(label),
            "order": torch.tensor(order).unsqueeze(0),
            "start_indices": torch.tensor([[con_start_idx, evt_start_idx]])
            if self.add_span_tokens
            else [],
            **tokenized,  # unpack tokenized input
        }
        return pair
