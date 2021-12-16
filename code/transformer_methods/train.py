import logging
import sys
import json
import os
import argparse

import pytorch_lightning as pl

from datetime import datetime
from pathlib import Path

from BioDataModule import BioDataModule
from CrossValDataModule import CrossValDataModule
from CrossValEnsembleDataModule import EnsembleCrossValDataModule
from SimpleTransformer import SimpleTransformer

logger = logging.getLogger("training")


def main(config, args):

    is_validation, k, fold = args.validation, args.k, args.fold

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=config["log_dir"],
        name=config["experiment_name"]+f"_k_{k}_fold_{fold}",
        default_hp_metric=False,
    )

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    logger.info(f"\n{json.dumps(config, indent=4, sort_keys=True)}")

    # Get commit of current code base
    stream = os.popen("git rev-parse --short HEAD")
    logger.info(f"Current git commit: {stream.read()}")

    pl.utilities.seed.seed_everything(config["run_config"]["seed"])

    if config["arch"]["ensemble"]["enabled"]:
        data = EnsembleCrossValDataModule(config, fold, k=k, all_data= True if is_validation else False)
        data.prepare_data()
    else:
        data = CrossValDataModule(config, fold, k=k, all_data= True if is_validation else False)
        data.prepare_data()

    model = SimpleTransformer(config)
    if config["run_config"]["pretrained_transformer"] != "":
        model.transformer = model.transformer.from_pretrained(
            config["run_config"]["pretrained_transformer"]
        )


    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor=config["early_stopping"]["metric"],
        min_delta=config["early_stopping"]["min_delta"],
        patience=config["early_stopping"]["patience"],
        mode=config["early_stopping"]["mode"],
        verbose=False,
    )

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=config["checkpoints"]["metric"],
        dirpath=config["checkpoints"]["save_dir"],
        filename=config["experiment_name"]+f"_k_{k}_fold_{fold}",
        mode=config["checkpoints"]["mode"],
    )

    test_type = config["run_config"]["test_type"]

    # Handle with care the configuration of the GPUs
    gpus_conf = dict()

    # If num-gpus is present, override which GPUs
    if args.num_gpus:
        num_gpus = args.num_gpus
        gpus_conf['gpus'] = num_gpus
        if num_gpus > 1:
            gpus_conf['accelerator'] = "ddp"
    # If the specific GPU is present and no explicit amount of gpus is specified, indicate it
    elif args.use_gpu:
        gpus_conf['gpus'] = [args.use_gpu]
        

    trainer = pl.Trainer(
        callbacks=[early_stop_callback, model_checkpoint_callback],
        # accelerator="ddp" if gpu else None,
        # gpus=[gpu] if gpu != None else None, 
        logger=tb_logger,
        accumulate_grad_batches=config["batch_accumulation_num"],
        max_epochs=config["max_epochs"],
        overfit_batches=5 if config["run_config"]["debug_run"] else 0,
        precision=config["float_precision"] if len(gpus_conf) > 0 else 32,
        progress_bar_refresh_rate=1 if config["progress_bar"] else 0,
        num_sanity_val_steps=2 if test_type != "test" else 0,
        check_val_every_n_epoch=1 if test_type != "test" else 10e5,  # 10e5 used as inf,
        ** gpus_conf # Expand the gpu parameters
    )

    if not args.test_ckpt:
        trainer.fit(model, data)
        if not is_validation:
            trainer.test(ckpt_path="best")
    else:
        log_dir = Path(tb_logger.save_dir, tb_logger.name)
        if not log_dir.exists():
            log_dir.mkdir()

        model = SimpleTransformer.load_from_checkpoint(checkpoint_path=args.test_ckpt, config=config)
        trainer.test(model, data)


if __name__ == "__main__":
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description="Train a simple transformer model")
    parser.add_argument("--config", type=str, default="test_conf.conf")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs to use if available")
    parser.add_argument("--use-gpu", type=int, default=None, help="Which GPUs to use")
    parser.add_argument("--fold", type=int, default=0, help="Fold number for CV")
    parser.add_argument("-k", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--test-ckpt", type=str, help="Run the test fold w/o training", default=None)
    parser.add_argument("--validation", action="store_true", help="Use this flag to do hyper parameter validation")
    args = parser.parse_args()
    assert os.path.isfile(args.config)
    with open(args.config, "r") as in_file:
        config = json.load(in_file)
    main(config, args)
    end_time = datetime.now()

    running_time = end_time - start_time
    print(f"\nTotal running time: {running_time}")
