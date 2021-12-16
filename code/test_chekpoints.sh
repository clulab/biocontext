#!/bin/bash

CKPT_DIR=$1
echo "Checkpoints dir: $CKPT_DIR"
for CKPT in "$CKPT_DIR"/*.ckpt; do
    # Get the ensemble type
    ENMSEMBLE=$(echo $(basename "$CKPT") | awk -F_ '{print $2}' -)
    # Resolve the conf file
    if [ $ENMSEMBLE == "ensemble" ];
    then
        CONF=confs/no_ensemble.conf
    elif [ $ENMSEMBLE == "post" ];
    then
        CONF=confs/post_distance.conf
    else
        CONF=confs/$ENMSEMBLE.conf
    fi

    
    # Get the fold num
    [[ "$(basename "$CKPT")" =~ fold_([0-9]) ]]
    FOLD="${BASH_REMATCH[1]}"

    # Execute the test script
    python transformer_methods/train.py --config "$CONF" --test-ckpt "$CKPT" --fold "$FOLD"
done