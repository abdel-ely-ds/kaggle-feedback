from typing import List
from collections import defaultdict
import pandas as pd

CLASSES = ['Lead',
 'Position',
 'Evidence',
 'Claim',
 'Concluding Statement',
 'Counterclaim',
 'Rebuttal']

MAX_LEN = 1024
STRIDE = 128

def label2index(classes: List[str] = CLASSES) -> dict:
    tags = defaultdict()

    for i, c in enumerate(classes):
        tags[f'B-{c}'] = i
        tags[f'I-{c}'] = i + len(classes)
    tags[f'O'] = len(classes) * 2
    tags[f'Special'] = -100

    l2i = dict(tags)

    return l2i

def index2label(l2i: dict) ->dict:

    i2l = defaultdict()
    for k, v in l2i.items():
        i2l[v] = k
    i2l[-100] = 'Special'

    i2l = dict(i2l)
    return i2l


def fix_beginnings(labels):
    for i in range(1,len(labels)):
        curr_lab = labels[i]
        prev_lab = labels[i-1]
        if curr_lab in range(7,14):
            if prev_lab != curr_lab and prev_lab != curr_lab - 7:
                labels[i] = curr_lab -7
    return labels


# tokenize and add labels
def tokenize_and_align_labels(examples,
                              tokenizer,
                              l2i,
                              stride: int = STRIDE,
                              max_length: int = MAX_LEN
                              ):
    o = tokenizer(examples['text'],
                  truncation=True,
                  padding=True,
                  return_offsets_mapping=True,
                  max_length=max_length,
                  stride=stride,
                  return_overflowing_tokens=True)

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = o["overflow_to_sample_mapping"]
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = o["offset_mapping"]

    o["labels"] = []

    for i in range(len(offset_mapping)):

        sample_index = sample_mapping[i]

        labels = [l2i['O'] for i in range(len(o['input_ids'][i]))]

        for label_start, label_end, label in \
                list(zip(examples['starts'][sample_index], examples['ends'][sample_index],
                         examples['classlist'][sample_index])):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start:
                    labels[j] = l2i[f'B-{label}']
                if token_start > label_start and token_end <= label_end:
                    labels[j] = l2i[f'I-{label}']

        for k, input_id in enumerate(o['input_ids'][i]):
            if input_id in [0, 1, 2]:
                labels[k] = -100

        labels = fix_beginnings(labels)

        o["labels"].append(labels)

    return o


def tokenize_for_validation(examples,
                            tokenizer,
                            l2i,
                            max_length):
    o = tokenizer(examples['text'], truncation=True, return_offsets_mapping=True, max_length=max_length)

    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = o["offset_mapping"]

    o["labels"] = []

    for i in range(len(offset_mapping)):

        labels = [l2i['O'] for i in range(len(o['input_ids'][i]))]

        for label_start, label_end, label in \
                list(zip(examples['starts'][i], examples['ends'][i], examples['classlist'][i])):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start:
                    labels[j] = l2i[f'B-{label}']
                if token_start > label_start and token_end <= label_end:
                    labels[j] = l2i[f'I-{label}']

        for k, input_id in enumerate(o['input_ids'][i]):
            if input_id in [0, 1, 2]:
                labels[k] = -100

        labels = fix_beginnings(labels)

        o["labels"].append(labels)

    return o



def get_truth_labels(tokenized_data, train_or_test="test") -> pd.DataFrame:
    l = []
    for example in tokenized_data[train_or_test]:
        for c, p in list(zip(example['classlist'], example['predictionstrings'])):
            l.append({
                'id': example['id'],
                'discourse_type': c,
                'predictionstring': p,
            })

    return pd.DataFrame(l)

