# The Classpose datasets use Numpy 2.0 which is incompatible with CellViT++
# (which uses Numpy ~1.24). So we first convert images.py and labels.py into
# npz files using Numpy 2.0 (npz are compatible with both Python versions)
# and then convert the npz files into npy files using the Numpy 1.24 which is
# used by uv.

DATASET_DIR=../datasets

for checkpoint in ../checkpoints/*.pth
do
    for dataset in $DATASET_DIR/classpose/monusac
    do
        DATASET_NAME=$(basename $dataset)
        MODEL_NAME=$(basename $checkpoint | cut -d '.' -f 1)
        echo Creating CellViT++ dataset for $DATASET_NAME with $MODEL_NAME
        out_dir=$DATASET_DIR/$DATASET_NAME
        python create-cellvitpp-dataset-1.py \
            --data_dir $dataset \
            --output_dir $out_dir \
            --sweep_name "$DATASET_NAME"_$MODEL_NAME \
            --checkpoint_path $checkpoint &&
        uv run python create-cellvitpp-dataset-2.py \
            --output_dir $out_dir &
    done
    wait
done

