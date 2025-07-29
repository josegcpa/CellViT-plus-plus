# The Classpose datasets use Numpy 2.0 which is incompatible with CellViT++
# (which uses Numpy ~1.24). So we first convert images.py and labels.py into
# npz files using Numpy 2.0 (npz are compatible with both Python versions)
# and then convert the npz files into npy files using the Numpy 1.24 which is
# used by uv.

DATASET_DIR=../datasets

for dataset in $DATASET_DIR/classpose/*
do
    DATASET_NAME=$(basename $dataset)
    echo Creating CellViT++ dataset for $DATASET_NAME
    out_dir=$DATASET_DIR/$DATASET_NAME
    rm -rf $out_dir
    python training_routine/create-cellvitpp-dataset-1.py \
        --data_dir $dataset \
        --output_dir $out_dir \
        --sweep_name $DATASET_NAME &&
    uv run python training_routine/create-cellvitpp-dataset-2.py \
        --output_dir $out_dir &
done

wait
