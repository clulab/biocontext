## Neural Architectures for Biological Inter-Sentence Relation Extraction

### Overview

This website hosts the code and the corpus associated to the [paper](tbd) presented at [SDU@AAAI 22](https://sites.google.com/view/sdu-aaai22/cfp?authuser=0)

### Corpus

The dataset used for this project is an extension of the dataset published by [Noriega et al. 2018](https://ml4ai.github.io/BioContext/). The original corpus contains hand curated annotations with text spans for biochemical events and biological context.

We extend this corpus with full text tokenized aligned to the original annotations to be make it compatible with neural network encoder architectures.

The data files can be downlodaded [here](https://github.com/clulab/neuralbiocontext/tree/gh-pages/corpus) and the parsing code is locatede [here](https://github.com/clulab/neuralbiocontext/blob/gh-pages/code/transformer_methods/BioDataset.py)

### Code and Instructions

The implementation of the neural architectures to detect biocontext can be found [here](https://github.com/clulab/neuralbiocontext/tree/gh-pages/code).

#### Instructions

To run the code, create the conda environment using the file named `conda_environment.yml`

Once the environment is created and configured, create a configuration file based on `test_conf.conf`. Replace the paths with your local environment's paths and configure the options and hyper parameters accordingly. The descriptions of the configuration fields can be found in `config_README.txt`.

To run the cross validation experiments described in this paper, edit the configuration file and run the following command from within the `code` directory. 
This will execute a training loop and testing run on fold `$FOLD` of cross validation, using the config file `$CONF_FILE`, using `$NUM_GPUS`, if available. Substitute the variables with your own values.

```bash
$ python transformer_methods/train.py --num-gpus $NUM_GPUS --fold $FOLD --conf $CONF_FILE 
```
To retrieve the testing scores (precision, recall and F1), run the command:

```bash
$ python transformer_methods/retrieve_cv_test_scores.py -i $EXP_PREFIX
```
Substitute the variable `$EXP_PREFIX` with the directory'a name prefix of the cross validation folds. For example, if you run the six-fold CV and have the following six directories with the tensorboard logs:

```
runs/experiment_cv_0/
runs/experiment_cv_1/
runs/experiment_cv_2/
runs/experiment_cv_3/
runs/experiment_cv_4/
runs/experiment_cv_5/
```
Substitute `$CONF_FILE` for `runs/experiment_cv_` to fetch the CV results.

### Citing

If you use this work or the full-text corpus, plese cite us using the following bibtex.

```
TBD
```

If you  use the annotations without relying on the full-text, please cite the authors using the following bibtex.

```bibtex
@ARTICLE{noriega-atala2020,

  author={Noriega-Atala, Enrique and Hein, Paul D. and Thumsi, Shraddha S. and Wong, Zechy and Wang, Xia and Hendryx, Sean M. and Morrison, Clayton T.},
  journal={IEEE/ACM Transactions on Computational Biology and Bioinformatics}, 
  title={Extracting Inter-Sentence Relations for Associating Biological Context with Events in Biomedical Texts}, 
  year={2020},
  volume={17},
  number={6},
  pages={1895-1906},
  doi={10.1109/TCBB.2019.2904231}}
  
}
```

