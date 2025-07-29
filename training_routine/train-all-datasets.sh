DATASET_DIR=../datasets
GPU=1

for dataset in $DATASET_DIR/classpose/*
do  
    DATASET_NAME=$(basename $dataset)
    DATA_DIR=$DATASET_DIR/$DATASET_NAME/train_configs/ViT256/fold_0.yaml
    echo Training CellViT++ for $DATASET_NAME
    uv run python ../cellvit/train_cell_classifier_head.py \
        --config $DATA_DIR --gpu $GPU --sweep
done
