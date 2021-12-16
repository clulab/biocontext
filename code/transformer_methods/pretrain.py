import logging
import sys
import json
import os
import argparse

import pytorch_lightning as pl

from pretraining.PretrainDataModule import PretrainDataModule
from pretraining.PretrainTransformer import PretrainTransformer
from CheckpointNSteps import CheckpointEveryNSteps

logger = logging.getLogger("training")


def main(config, gpu):
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=config["log_dir"],
        name=config["experiment_name"],
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

    data = PretrainDataModule(config)
    data.prepare_data()
    data.setup(stage="fit")

    model = PretrainTransformer(config, data.num_event_types)

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
        filename=config["experiment_name"],
        mode=config["checkpoints"]["mode"],
    )

    step_checkpoint_callback = CheckpointEveryNSteps(save_step_frequency=10000)

    test_type = config["run_config"]["test_type"]
    trainer = pl.Trainer(
        callbacks=[
            early_stop_callback,
            model_checkpoint_callback,
            step_checkpoint_callback,
        ],
        gpus=[gpu] if gpu != None else None,
        logger=tb_logger,
        log_every_n_steps=100,
        accumulate_grad_batches=config["batch_accumulation_num"],
        max_epochs=config["max_epochs"],
        overfit_batches=5 if config["run_config"]["debug_run"] else 0,
        precision=config["float_precision"],
        progress_bar_refresh_rate=1 if config["progress_bar"] else 0,
        num_sanity_val_steps=2 if test_type != "test" else 0,
        check_val_every_n_epoch=1 if test_type != "test" else 10e5,  # 10e5 used as inf
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain a transformer model")
    parser.add_argument("--config", type=str)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    assert os.path.isfile(args.config)
    with open(args.config, "r") as in_file:
        config = json.load(in_file)
    main(config, args.gpu)
