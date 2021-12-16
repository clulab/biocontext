import itertools
import plac
import csv
import json
from pathlib import Path
import itertools as it
from CrossValDataModule import CrossValDataModule

@plac.pos('conf_path', type=Path)
@plac.pos('k', type=int, help='Number of folds')
def main(conf_path: Path = Path("test_conf.conf"), k: int = 5):
    
    with open(conf_path, 'r') as f:
        conf = json.load(f)

    data_module = CrossValDataModule(conf, fold_idx=0, k= k) # fold_ix is irrelevant here
    data_module.prepare_data()

    # Get the papers for each fold
    splits = dict()
    for fold_ix in range(k+1):
        dataset = data_module.dataset
        data_module.get_fold_papers
        train_papers, train_names, val_papers, val_names, test_papers, test_names = data_module.get_fold_papers(fold_ix)

        test_pairs, train_pairs, val_pairs = {}, {}, {}
        train_pairs['positive'] = list(itertools.chain.from_iterable(dataset.paper_pairs[p]['positive'] for p in train_names))
        train_pairs['negative'] = list(itertools.chain.from_iterable(dataset.paper_pairs[p]['negative'] for p in train_names))
        test_pairs['positive'] = list(itertools.chain.from_iterable(dataset.paper_pairs[p]['positive'] for p in test_names))
        test_pairs['negative'] = list(itertools.chain.from_iterable(dataset.paper_pairs[p]['negative'] for p in test_names))
        val_pairs['positive'] = list(itertools.chain.from_iterable(dataset.paper_pairs[p]['positive'] for p in val_names))
        val_pairs['negative'] = list(itertools.chain.from_iterable(dataset.paper_pairs[p]['negative'] for p in val_names))

        assert len(set(test_pairs['negative']) & set(test_pairs['positive'])) == 0

        splits[fold_ix] = {
            'train': train_names,
            'val': val_names,
            'test': test_names,
            'test_pairs': test_pairs,
            'train_pairs': train_pairs,
            'val_pairs': val_pairs
        }

    print(json.dumps(splits, indent=4))

    negative_pairs = set(it.chain.from_iterable(s['test_pairs']['negative'] + s['val_pairs']['negative'] for s in splits.values()))

    with open('negative_pairs.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(negative_pairs)


if __name__ == '__main__':
    plac.call(main)
    
