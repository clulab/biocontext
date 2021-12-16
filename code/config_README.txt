{
    "experiment_name": "dev",
        ~ Name for experiment in tensorboard logs

    "data_dir": "/work/plovett/data/papers/event_spans_Reach2016/",
        ~ Directory containing subdirectories for each paper in the dataset

    "log_dir": "lightning_logs",
        ~ Name of directory to store the tensorboard logs in

    "hf_model_name": "allenai/biomed_roberta_base",
        ~ Name of the hugging face model to use as the transformer

    "early_stopping": {
        "metric": "validation_f1",
            ~ Which metric logged to tensorboard to track

        "min_delta": 0.001,
            ~ Threshold for detecting differences between scores

        "patience": 3,
            ~ How many epochs to wait for a better score

        "mode": "max"
            ~ How to judge improvements in score. Is a higher or lower score
              better?
    },
    "checkpoints": {
        "metric": "validation_f1",
            ~ Which metric logged to tensorboard to track

        "save_dir": "./",
            ~ Which directory to store checkpoint files in 

        "mode": "max"
            ~ How to judge improvements in score. Is a higher or lower score
              better?
    },
    "run_config": {
        "test_type": "validation",
            ~ Controls whether we evaluate on test data or validation data.
              There is currently no implementation for test runs.

        "test_paper_index": 0,
            ~ Which paper forms is the first of the 4 papers used as
              validation. If 0 is chosen then [0, 1, 2, 3] will be the
              validation papers. If 3 is chosen then [3, 4, 5, 6] etc.

        "validation_paper_count": 5,
            ~ How many papers to include in the validation sample

        "seed": 123,
            ~ Seed for random number generators.

        "debug_run": true,
            ~ If debug is activated, the model will overfit to a small number
              of training examples. 

        "pretrained_transformer": "saved/step-checkpoint_epoch-2_step-40000.roberta"
            ~ If this is a non-empty string, the model will load the 
              transformer weights from the filepath provided.
    },
    "hyperparams": {
        "hidden_layer_width": 100,
            ~ This controls the size of the final feed forward layer.

        "learning_rate": 3e-5,
            ~ Self explantory

        "negative_sample_weight": 0.5
            ~ Controls the weighting of negative samples in the loss function.
    
    },
    "arch": {
        "order_linear_proj_enabled": false,
            ~ If this is set to True, the model will conditionally activate
              feed forward weights depending on if the context comes before 
              the event or after. 

        "add_span_tokens": true,
            ~ If this is set to True, the <CON_START>, <CON_END>, <EVT_START>,
              and <EVT_END> tags will be placed around the context and events.

        "use_span_not_cls": true,
            ~ Controls whether to use the <CON_START> and <EVT_START> tags or 
              to use the <CLS> tag as the final representation.

        "ensemble": {
            "enabled": true,
                ~ Whether or not to enable an ensemble model.

            "type": "vote"
                ~ Controls the type of ensemble. Acceptable values are:
                  'vote' and 'average'
        }
    },
    "max_epochs": 20,
        ~ Sets the upper limit for epochs.

    "batch_size": 1,
        ~ Sets the training batch size. Validation and test batches are 
          hard coded to 1.

    "batch_accumulation_num": 1,
        ~ Controls the accumulation of loss across batches, enabling
          effective batch sizes to be a multiple of the above value.

    "train_thread_count": 0,
        ~ Controls the number of threads used in the training dataloader.

    "eval_thread_count": 0,
        ~ Controls the number of threads used in the val & test dataloader.

    "float_precision": 32,
        ~ Allows the system to use lower precision, which fits larger batches
          into GPU memory. 

    "progress_bar": true
        ~ Enables a progress bar. Do not enable on HPC, it will fill the 
          output with half-printed loading bars.
}
