DATASET_DIR=../datasets
GPU=1

for dataset in $DATASET_DIR/classpose/*
do  
    DATASET_NAME=$(basename $dataset)
    for model in $DATASET_DIR/$DATASET_NAME/train_configs/*
    do 
        CONFIG_FILE=$model/fold_0.yaml
        echo Training CellViT++ for $DATASET_NAME with $(basename $model)
        uv run python ../cellvit/train_cell_classifier_head.py \
            --config $CONFIG_FILE --gpu $GPU --sweep
    done
done
