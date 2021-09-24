BASE_PATH=/livingrooms/wiz94156/LibriSpeech-Feature/
for SEED in 57 666 1450
do

    OUTPUT_PATH="/groups/wiz94156/Privacy-issues/MIA/fake-500uttr-learnable/seed-${SEED}"
    # hubert wav2vec2 modified_cpc tera 
    for MODEL in hubert wav2vec2 modified_cpc tera 
    do
        echo "[running seed-${SEED}]"
        echo "[running utterance-level learnable attack]"
        echo "[running ${MODEL}]"

        SIM_MODEL_PATH="/groups/wiz94156/Privacy-issues/MIA/fake-500uttr-learnable/similarity-model-seed-${SEED}/learnable-similarity-${MODEL}.pt"

        echo "[attacking...]"
        python utterance-level-MIA-learnable.py \
            --base_path $BASE_PATH \
            --output_path $OUTPUT_PATH \
            --sim_model_path $SIM_MODEL_PATH \
            --model $MODEL \
            --seed $SEED \
            --batch_size 200 \
            --num_workers 2

    done
done
