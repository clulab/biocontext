from os import error
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf
import itertools as it
import plac
import glob
import csv
from pathlib import Path

import numpy as np

tf.get_logger().setLevel('INFO')

def get_tensorboard_scores(g):
    # Get the validation scores
    
    f1_p, precision_p, recall_p, support_p = 0, 0, 0, 0
    f1_n, precision_n, recall_n, support_n = 0, 0, 0, 0
    labels, predictions, data_ids = "x", "x", "x"
    g = list(g)

    for e in g:
        for v in e.summary.value:
            if v.tag == 'test_f1_positive':
                f1_p = v.simple_value
            elif v.tag == 'test_precision_positive':
                precision_p = v.simple_value
            elif v.tag == 'test_recall_positive':
                recall_p = v.simple_value
            elif v.tag == 'test_support_positive':
                support_p = v.simple_value
            elif v.tag == 'test_f1_negative':
                f1_n = v.simple_value
            elif v.tag == 'test_precision_negative':
                precision_n = v.simple_value
            elif v.tag == 'test_recall_negative':
                recall_n = v.simple_value
            elif v.tag == 'test_support_negative':
                support_n = v.simple_value
            elif v.tag == 'test_predictions/text_summary':
                predictions = v.tensor.string_val
            elif v.tag == 'test_labels/text_summary':
                labels = v.tensor.string_val
            elif v.tag == "test_data_ids/text_summary":
                data_ids = v.tensor.string_val
            

    f1, precision, recall = np.asarray([f1_n, f1_p]).mean(), np.asarray([precision_n, precision_p]).mean(), np.asarray([recall_n, recall_p]).mean()

    return f1_n, precision_n, recall_n, support_n, f1_p, precision_p, recall_p, support_p,  f1, precision, recall, labels, predictions, data_ids

@plac.pos("exp_prefix", "Prefix of the KF-CV directories", type=str)
@plac.flg("is_crossval", "Wether this is a crosval result")
def main(exp_prefix: str, is_crossval: bool = False):

    # Check to see if this is a CV prefix, otherwise just use it as stand alone directory

    frame_data = list()
    
    if is_crossval:
        # Get the number of folds
        dirs = list(sorted(glob.glob(exp_prefix + "*_k_*_fold_*")))
        print(dirs)
        test_dir = Path(dirs[0])
        k = int(test_dir.name.split("_")[-3])
        # Get the list of cv directories
        dirs = [Path(d) for d in dirs]
    else:
        dirs = [p for p in Path(exp_prefix).iterdir() if p.is_dir()]


    all_predictions, all_labels, all_ids = list(), list(), list()

    
    for ix, log_dir in enumerate(dirs):
        
        best_f1, best_precision, best_recall = 0, 0, 0
        best_f1_p, best_precision_p, best_recall_p, best_support_p = 0, 0, 0, 0
        best_f1_n, best_precision_n, best_recall_n, best_support_n = 0, 0, 0, 0
        best_labels, best_predictions, best_data_ids = "", "", ""

        best_exp_name = None
        missing_folds = list()
        missing = True
        for log_file in log_dir.rglob('events.out*'):
            exp_name  = str(log_file)
            data = list(tf.compat.v1.train.summary_iterator(str(log_file)))
            
            f1_n, p_n, r_n, s_n, f1_p, p_p, r_p, s_p, f1, p, r, labels, preds, data_ids = get_tensorboard_scores(data)
            if f1_p > best_f1_p:
                best_f1, best_precision, best_recall = f1, p, r
                best_f1_p, best_precision_p, best_recall_p, best_support_p = f1_p, p_p, r_p, s_p
                best_f1_n, best_precision_n, best_recall_n, best_support_n = f1_n, p_n, r_n, s_n
                best_labels, best_predictions, best_data_ids = labels, preds, data_ids
                best_exp_name = exp_name
                missing = False

        config_name = log_dir.name.split("_cv")[0]
        if missing:
            missing_folds.append(str(ix))
        else:
            if is_crossval:
                fold = int(log_dir.name.split("_")[-1])
                print(f"{config_name} Fold: {fold}")

            print(f'Best testing scores: F1: {best_f1_p}\tP:{best_precision_p}\tR:{best_recall_p}')
            print(f'Exp: {best_exp_name}')

        
        if is_crossval:
            
            row = {'conf': config_name,
            'fold':int(log_dir.name.split("_")[-1]),
            'p':best_precision if best_precision > 0 else '',
            'r':best_recall if best_recall > 0 else '',
            'f1':best_f1 if best_f1 > 0 else '',
            'p_n':best_precision_n if best_precision_n > 0 else '',
            'r_n':best_recall_n if best_recall_n > 0 else '',
            'f1_n':best_f1_n if best_f1_n > 0 else '',
            's_n':best_support_n if best_support_n > 0 else '',
            'p_p':best_precision_p if best_precision_p > 0 else '',
            'r_p':best_recall_p if best_recall_p > 0 else '',
            'f1_p':best_f1_p if best_f1_p > 0 else '',
            's_p':best_support_p if best_support_p > 0 else '',
            }
            frame_data.append(row)

        try:
            all_predictions.extend(best_predictions[0].decode().split('\n'))
            all_labels.extend(best_labels[0].decode().split('\n'))
            all_ids.extend(best_data_ids[0].decode().split('\n'))
        except AttributeError:
            x = 0


    if is_crossval:
        
        with open('table.csv', 'w') as f:
            import csv
            w = csv.DictWriter(f, fieldnames=frame_data[0].keys())
            w.writeheader()
            w.writerows(frame_data)

        predictions_file = Path(f'{exp_prefix}_predictions.tsv')
        save_raw_predictions(predictions_file.name, all_labels, all_predictions, all_ids)

def save_raw_predictions(filename, y_true, y_pred, data_ids):

    assert len(y_true) == len(y_pred) == len(data_ids), "Unequal sized inputs"

    data_ids = [d.replace('\t', '|') for d in data_ids]

    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['y_true', 'y_pred', 'data_id'])
        writer.writerows(zip(y_true, y_pred, data_ids))

if __name__ == "__main__":
    plac.call(main)