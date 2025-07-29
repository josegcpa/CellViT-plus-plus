set -e

SWEEPS_DIR=../logs_local
CELLVIT_PATH=../checkpoints/CellViT-Virchow-x40-AMP.pth

for sweep in $SWEEPS_DIR/sweep_*
do
    BEST_CONFIGURATION=$(dirname $(
        uv run python3 ../scripts/find_best_hyperparameter.py \
            $sweep \
            --metric AUROC/Validation | grep config.yaml))
    
    DATASET_PATH=$(cat $BEST_CONFIGURATION/config.yaml | 
        grep dataset_path: | 
        tr -d ' ' | 
        sed 's/dataset_path://'
    )
    echo $DATASET_PATH
    identify -format "%f,%h,%w\n" $DATASET_PATH/test/images/*png > $DATASET_PATH/image_sizes.txt
    MAX_HEIGHT=$(sort -nr $DATASET_PATH/image_sizes.txt | head -n1 | cut -d',' -f2)
    MAX_WIDTH=$(sort -nr $DATASET_PATH/image_sizes.txt | head -n1 | cut -d',' -f3)

    echo Evaluating $sweep
    echo "- Using $BEST_CONFIGURATION"
    echo "- Using $DATASET_PATH"
    echo "- Using $CELLVIT_PATH"
    echo "- Original image size: $MAX_HEIGHT x $MAX_WIDTH"

    if [[ $MAX_HEIGHT -gt 256 || $MAX_WIDTH -gt 256 ]]; then
        MAX_HEIGHT=1024
        MAX_WIDTH=1024
    else
        MAX_HEIGHT=256
        MAX_WIDTH=256
    fi

    echo "- Using $MAX_HEIGHT x $MAX_WIDTH"

    echo uv run python3 ../cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
        --logdir $BEST_CONFIGURATION \
        --dataset_path $DATASET_PATH \
        --cellvit_path $CELLVIT_PATH \
        --input_shape $MAX_HEIGHT $MAX_WIDTH
    
    # send this to the background
    uv run python calculate-metrics.py \
        --logdir $BEST_CONFIGURATION \
        --output_path $sweep &
    
done