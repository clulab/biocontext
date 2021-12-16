import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
import itertools as it
import plac
from pathlib import Path

tf.get_logger().setLevel('INFO')

def get_validation_scores(g):
    # Get the validation scores
    f1, precision, recall = 0, 0, 0
    for e in g:
        for v in e.summary.value:
            if v.tag == 'validation_f1':
                f1 = v.simple_value
            elif v.tag == 'validation_precision':
                precision = v.simple_value
            elif v.tag == 'validation_recall':
                recall = v.simple_value
    return f1, precision, recall

def get_best_validation_scores(data):
    # First group the entries by their walltime
    groups = it.groupby(data, lambda x: int(x.wall_time))

    # Now iterate over the groups and find the best validation score
    best_f1, best_precision, best_recall = 0, 0, 0
    for wall_time, group in groups:
        # Get the validation scores
        f1, precision, recall = get_validation_scores(group)

        # Update the best scores
        if f1 > best_f1:
            best_f1, best_precision, best_recall = f1, precision, recall

    return best_f1, best_precision, best_recall

@plac.pos("log_dir", "The log directory", type=Path)
def main(log_dir: Path):

    best_f1, best_precision, best_recall = 0, 0, 0
    best_exp_name = None

    for log_file in Path(log_dir).rglob('events.out*'):
        exp_name  = str(log_file)
        data = tf.compat.v1.train.summary_iterator(str(log_file))
        f1, p, r = get_best_validation_scores(data)
        if f1 > best_f1:
            best_f1, best_precision, best_recall = f1, p, r
            best_exp_name = exp_name

    print(f'Best validation scores: F1: {best_f1}\tP:{best_precision}\tR:{best_recall}')
    print(f'Exp: {best_exp_name}')


if __name__ == "__main__":
    plac.call(main)