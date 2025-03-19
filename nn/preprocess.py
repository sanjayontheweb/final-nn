# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import pandas as pd

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    sequences  = pd.DataFrame({"seqs": seqs, "labels": labels})
    pos_seqs = sequences[sequences["labels"] == True]
    neg_seqs = sequences[sequences["labels"] == False]

    if len(pos_seqs) > len(neg_seqs):
        neg_seqs = neg_seqs.sample(len(pos_seqs), replace=True)
    else:
        pos_seqs = pos_seqs.sample(len(neg_seqs), replace=True)
    
    sampled_seqs = pd.concat([pos_seqs, neg_seqs])
    sampled_seqs = sampled_seqs.sample(frac=1).reset_index(drop=True)

    return sampled_seqs["seqs"].tolist(), sampled_seqs["labels"].tolist()


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    one_hot_dict = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    encodings = []

    for seq in seq_arr:
        seq_encoding = []
        for nucleotide in seq:
            seq_encoding.extend(one_hot_dict[nucleotide])
        encodings.append(seq_encoding)

    return np.array(encodings)