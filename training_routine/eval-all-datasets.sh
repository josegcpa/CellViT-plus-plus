set -e

SWEEPS_DIR=logs_local
METRICS_DIR=metrics

mkdir -p $METRICS_DIR

for sweep in $SWEEPS_DIR/sweep_*
do
    DATASET_NAME=$(echo $sweep | cut -d _ -f 3)
    checkpoint=$(ls ../checkpoints/$(echo $sweep | cut -d '_' -f 4)*pth)
    CKPT_NAME=$(basename $checkpoint | cut -d '.' -f 1)
    OUT_METRICS_FILE=$METRICS_DIR/"$DATASET_NAME"_$CKPT_NAME.csv
    OUT_METRICS_FILE_CELLVIT=$METRICS_DIR/"$DATASET_NAME"_"$CKPT_NAME"_cellvit.csv

    if [ -f $OUT_METRICS_FILE ] && [ -f $OUT_METRICS_FILE_CELLVIT ]; then
        echo "Skipping $sweep"
        # continue
    fi

    BEST_CONFIGURATION=$(dirname $(
        uv run python3 ../scripts/find_best_hyperparameter.py \
            $sweep \
            --metric AUROC/Validation | grep config.yaml))
    
    DATASET_PATH=$(cat $BEST_CONFIGURATION/config.yaml | 
        grep dataset_path: | 
        tr -d ' ' | 
        sed 's/dataset_path://'
    )

    identify -format "%f,%h,%w\n" $DATASET_PATH/test/images/*png > $DATASET_PATH/image_sizes.txt
    MAX_HEIGHT=$(sort -nr $DATASET_PATH/image_sizes.txt | head -n1 | cut -d',' -f2)
    MAX_WIDTH=$(sort -nr $DATASET_PATH/image_sizes.txt | head -n1 | cut -d',' -f3)

    echo Evaluating $sweep
    echo "- Using $BEST_CONFIGURATION"
    echo "- Using $DATASET_PATH"
    echo "- Using $checkpoint"
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
        --cellvit_path $checkpoint \
        --input_shape $MAX_HEIGHT $MAX_WIDTH
    
    echo Calculating metrics

    uv run python calculate-metrics.py \
        --logdir $BEST_CONFIGURATION \
        --output_path $sweep
    
    cp $sweep/metrics.csv $OUT_METRICS_FILE
    cp $sweep/metrics_cellvit.csv $OUT_METRICS_FILE_CELLVIT
    cp $sweep/predictions.npy $METRICS_DIR/"$DATASET_NAME"_"$CKPT_NAME"_predictions.npy
    cp $sweep/labels.npy $METRICS_DIR/"$DATASET_NAME"_"$CKPT_NAME"_labels.npy
done