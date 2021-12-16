import os
import logging
import itertools
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


logger = logging.getLogger("training")


class PretrainDataset(Dataset):
    """
    Dataset for pretraining on papers which have had events and contexts
    automatically recognized by Reach.
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
        tokenizer_name: str
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

        if not os.path.isdir(data_dir):
            logger.error(f"ERROR: Provided data_dir path is not a directory.")
            raise FileExistsError

        paper_paths = []
        for folder in os.listdir(data_dir):
            paper_paths.append(folder)

        if len(paper_paths) == 0:
            logger.error(f"ERROR: No paper dirs found at path: {data_dir}")
            raise FileNotFoundError

        # Use multiprocessing to speed up paper preprocessing as each paper
        # is independent at this stage.
        with ProcessPoolExecutor(max_workers=20) as executor:
            self.paper_data = list(executor.map(
                get_paper_data, itertools.repeat(data_dir), paper_paths, chunksize=20
            ))
            
        to_be_removed = []
        rule_type_index = {}
        self.number_of_pairs = 0
        # Remove all papers that errored out or had no events. 
        for i, paper in enumerate(self.paper_data):
            if len(paper) == 0 or len(paper["events"]) == 0:
                to_be_removed.append(i)
            else:
                for event in paper["events"]:
                    if event["rule_type"] not in rule_type_index:
                        rule_type_index[event["rule_type"]] = len(rule_type_index)
                self.number_of_pairs += paper["num_pairs"]

        # Order is reversed so that we don't have to recalculate indices after
        # each deletion.
        for idx in sorted(to_be_removed, reverse=True):
            self.paper_data.pop(idx)

        self.event_type_index = rule_type_index
        self.num_event_types = len(rule_type_index)
        self.num_papers = len(self.paper_data)

        # Sort papers for consistent indexing. We want the original indexing to
        # be consistent so that we can control the randomizing of our indexing
        # later using random seeds.
        self.paper_data.sort(key=lambda x: int(x["name"]))

        # Generate folds (or per paper indices)
        # i.e. each paper covers a section of the total number of indices
        #  we want a map from paper -> subsection of indices so that we
        #  can easily create held out subsets at the paper level
        self.last_indice = []
        prev_end = 0
        for paper in self.paper_data:
            # For every event choose 2 unique contexts
            prev_end += len(paper["events"]) * \
                len(paper["context_index"]) * \
                (len(paper["context_index"])-1)
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
        # Samples are defined by the following ordering
        # Paper | Event | Context_a | Context_b
        # So if there are n papers, m_i events, and c_i contexts for paper i
        # then there are: sum{0, n, i}(m_i * c_i * (c_i-1)) pairings.
        # The data samples are organized as follows:
        # event_1 x context_1 x context_2, ..., event_1 x context_c x context_c-1,
        # event_2 x context_1 x context_2, ... until event_n x context_c after 
        # which we move onto the next paper
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

        # For each event there is a pair of two context types
        # i.e. num_event x num_context x (num_context-1) | modulo arithmetic 
        # lets us recover indicies:
        #  event_idx * num_context * (num_context-1) + context_idx
        num_context_pairs = len(paper["context_index"]) * (len(paper["context_index"])-1)
        event_idx = index // num_context_pairs
        context_pair_idx = index % num_context_pairs
        context_a_idx = context_pair_idx // (len(paper["context_index"]) - 1)
        context_b_idx = context_pair_idx % (len(paper["context_index"]) - 1)
        
        # Adjust to account for us skipping a_idx == b_idx
        #  all indicies after a_idx have to be adjusted by 1
        if context_b_idx >= context_a_idx:
            context_b_idx += 1

        try:
            event = paper["events"][event_idx]
        except Exception:
            print(f"index passed in: {index}")
            print(f"final paper index: {paper_index}")
            print(f"paper_index_start: {paper_index_start}")
            raise Exception

        # Get the context mentions for both context ids. Also get the range of 
        # sentence indices between the event and the closest context mention.
        context_mentions_a, spans_a = event["context_pairs"][context_a_idx]
        context_mentions_b, spans_b = event["context_pairs"][context_b_idx]
        context_mentions = context_mentions_a + context_mentions_b
        spans = spans_a + spans_b
        
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
            # First two contexts are type a and second two are type b
            if i <= 1: 
                context_idx = context_a_idx
            else:
                context_idx = context_b_idx

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
        for mention, sentence_span in zip(context_mentions, sentence_spans):
            # Find out if context is before or after event
            # Get the indices for each sentence. They should be either the
            # first or the last sentence in the spans.
            if event["sentence"] < mention["sentence"]:
                orders.append([0, 1])
                evt_sent = sentence_span[0]
                con_sent = sentence_span[-1]
                evt_idx = 0
                con_idx = -1
            else:
                orders.append([1, 0])
                evt_sent = sentence_span[-1]
                con_sent = sentence_span[0]
                evt_idx = -1
                con_idx = 0

            # Calculate indices of span starts
            evt_s, evt_e = event["start"], event["end"]
            con_s, con_e = mention["start"], mention["end"]
            if event["sentence"] == mention["sentence"]:
                # If the sentence indices are equal, then sentence_span 
                # should be a single sentence
                try:
                    assert len(sentence_span) == 1
                except:
                    print(f"Sentence span: {sentence_span}")
                    print(f"Paper: {paper['name']}")
                    print(f"Context: {mention['context_name']}")
                    print(f"Event: {event['rule_type']}")
                    print(f"Event: {evt_s} - {evt_e}")
                    print(f"Context: {con_s} - {con_e}")
                    raise 
                evt_start_idx = evt_s
                con_start_idx = con_s
                
                # add start / end tokens to sentences
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
            try:
                assert tokenized["input_ids"].shape[1] == 512
            except:
                print(f"Sentence span: {sentence_span}")
                print(f"Tokenized shape: {tokenized['input_ids'].shape}")
                print(f"Paper: {paper['name']}")
                print(f"Context: {mention['context_name']}")
                print(f"Event: {event['rule_type']}")
                raise 
            start_end_pairs.append([con_start_idx, evt_start_idx])
            tokenized_outputs.append(tokenized)

        # The event stays constant so all of the samples have the same event label.
        labels = [self.event_type_index[event['rule_type']] for _ in orders]
        pair = {
            "name": paper["name"],
            "labels": torch.tensor(labels),
            "order": torch.tensor(orders),
            "start_indices": torch.tensor(start_end_pairs),
            "tokenizer_output": tokenized_outputs,
        }
        return pair


def get_paper_data(data_dir, paper):
    """
    Takes in a directory and a directory within it and collects all the
    necessary information about the paper in that directory. Returns 
    information as dictionary to be later collated.
    """
    
    sentence_path = os.path.join(data_dir, paper, "sentences.txt")
    if not os.path.exists(sentence_path):
        logger.warning(
            f"No sentence file found for paper dir: {paper} skipping..."
        )
        return {}

    mention_path = os.path.join(data_dir, paper, "mention_intervals.txt")
    if not os.path.exists(mention_path):
        logger.warning(
            f"No context mention file found for paper dir: {paper} skipping..."
        )
        return {}

    event_path = os.path.join(data_dir, paper, "event_intervals.txt")
    if not os.path.exists(event_path):
        logger.warning(
            f"No event file found for paper dir: {paper} skipping..."
        )
        return {}

    raw_sentences = []
    with open(sentence_path, "r") as in_file:
        # Sentences should already be pre-tokenized
        for line in in_file:
            cur = line.strip().split()
            raw_sentences.append(cur)

    print(f"Number of sentences: {len(raw_sentences)}")

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
            if len(cur) <= 1:
                continue

            sent_idx = int(cur[0])
            for context in cur[1:]:
                context = context.replace("%%", "%")
                temp = context.split("%")
                sent = raw_sentences[sent_idx]
                start, end = int(temp[0]), int(temp[1])
                try:
                    context_name = temp[3]
                except:
                    raise Exception(f"Bad line in file: {paper}")
                if context_name == "":
                    continue

                end += 1  # add one to convert from [x, y] to [x, y)
                try:    
                    assert start >= 0 and start < len(sent)
                    assert end >= 0 and end <= len(sent)
                except:
                    print(temp)
                    print(f"start: {start}")
                    print(f"end: {end}")
                    print(f"span: {temp[2]}")
                    print(f"sent[start:end]: {sent[start:end]}")
                    print(f"sent len: {len(sent)}")
                    print(f"sent: {sent}")
                    print(f"sent idx: {sent_idx}")
                    raise


                new_mention = {}
                new_mention["sentence"] = sent_idx  # int
                new_mention["start"] = start  # int
                new_mention["end"] = end  # int
                new_mention["context_name"] = context_name  # str

                # Update reverse index before adding mention to list
                if context_name in context_index:
                    context_id = context_index[context_name]
                    contexts[context_id].append(new_mention)
                else:
                    context_index[context_name] = len(context_index)
                    context_id = context_index[context_name]
                    contexts[context_id] = [new_mention]

                # add context mention to reverse index
                #  format is tuple (context id, index in contexts)
                if sent_idx not in sentence_content:
                    sentence_content[sent_idx] = ([], [])
                sentence_content[sent_idx][PretrainDataset.CON].append(
                    (context_id, len(contexts[context_id]) - 1)
                )

    # Remove all contexts with fewer than 2 mentions
    to_be_deleted = []
    for context_id in contexts:
        if len(contexts[context_id]) >= 2:
            continue

        to_be_deleted.append(context_id)

        for sent_idx in sentence_content:
            indicies_to_remove = []
            for i, mention in enumerate(sentence_content[sent_idx][PretrainDataset.CON]):
                if mention[0] == context_id:
                    indicies_to_remove.append(i)

            for idx in indicies_to_remove:
                sentence_content[sent_idx][PretrainDataset.CON].pop(idx)

        for name in context_index:
            if context_index[name] == context_id:
                context_name = name
                break
            
        if context_name == "":
            print(context_index)
            raise Exception(
                f"Couldn't find context id: {context_id} in context index of size: {len(context_index)}"
            )
        del context_index[context_name]

    for context_id in to_be_deleted:
        del contexts[context_id]

    # Now reset context index to be 0-n for n contexts
    new_contexts = {}
    old2new = {}
    i = 0
    for key in sorted(list(contexts)):
        old2new[key] = i
        new_contexts[i] = contexts[key]
        i += 1
    contexts = new_contexts

    # Now fix indices inside sentence_content
    for sent_idx in sentence_content:
        new_context_content = []
        for mention in sentence_content[sent_idx][PretrainDataset.CON]:
            new_context_content.append((old2new[mention[0]], mention[1]))
        sentence_content[sent_idx] = (new_context_content, [])

    # Collect all of the events
    events = []
    with open(event_path, "r") as in_file:
        # Each line should be:
        # sentence_index \t start_token-end_token-rule_used
        for line in in_file:
            cur = line.strip().split()
            if len(cur) <= 1:
                continue

            sent_idx = int(cur[0])
            for event in cur[1:]:
                start, end, rule_type = event.split("-")
                start, end = int(start), int(end)

                end += 1  # add one to convert from [x, y] to [x, y)
                try:
                    assert start >= 0 and start < len(raw_sentences[sent_idx])
                    assert end >= 0 and end <= len(raw_sentences[sent_idx])
                except:
                    print(event)
                    print(f"start: {start}")
                    print(f"end: {end}")
                    print(f"sent[start:end]: {raw_sentences[sent_idx][start:end]}")
                    print(f"sent len: {len(raw_sentences[sent_idx])}")
                    print(f"sent: {raw_sentences[sent_idx]}")
                    print(f"sent idx: {sent_idx}")
                    raise

                new_event = {}
                new_event["sentence"] = sent_idx  # int
                new_event["start"] = start  # int
                new_event["end"] = end  # int
                new_event["rule_type"] = rule_type  # str

                # For each context, sort the sentence spans by distance to 
                # the event
                spans = dict()
                for context_id in contexts:
                    context_mentions = sorted(
                        contexts[context_id],
                        key=lambda x: (new_event["sentence"] - x["sentence"])
                        ** 2,
                    )[:2] # Just keep the two closest mentions

                    span_list = []
                    for context in context_mentions:
                        # Get all sentences between context and event
                        # Have to add one because range is [x, y)
                        if context["sentence"] > new_event["sentence"]: 
                            # This case captures when they are equal
                            span_list.append(
                                range(
                                    new_event["sentence"],
                                    context["sentence"] + 1,
                                )
                            )
                        else: 
                            span_list.append(
                                range(
                                    context["sentence"],
                                    new_event["sentence"] + 1,
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
            sentence_content[sent_idx][PretrainDataset.EVT].append(len(events) - 1)

    # The number of potential samples in the paper is for every event
    # how many combinations of two context mentions can be calculated
    number_of_pairs = len(events) * len(context_index) * (len(context_index)-1)
    cur_paper = {
        "name": paper,  # str
        "sentences": raw_sentences,  # List[List[str]]
        "events": events,  # List[Dict[str, obj]]
        "context_index": context_index,  # Dict[str, int]
        "contexts": contexts,  # Dict[int, List[obj]]
        "num_pairs": number_of_pairs, # int
        "sentence_content": sentence_content,  # Dict[Tuple(List[int], List[int, int])]
    }
    return cur_paper
