BASE_PATH=/livingrooms/wiz94156/LibriSpeech-Feature/

#  hubert_large_ll60k hubert_xtralarge_ll60k wav2vec2_large_960 wav2vec2_large_ll60k tera_100hr
for SEED in 57 666 1450
do
    OUTPUT_PATH="/groups/wiz94156/Privacy-issues/MIA/fake-1spkr-learnable/similarity-model-seed-${SEED}"
    
    for MODEL in hubert_large_ll60k hubert_xtralarge_ll60k wav2vec2_large_960 wav2vec2_large_ll60k tera_100hr
    do
        echo "[running seed ${SEED}]"
        echo "[training speaker-level similarity model for ${MODEL}...]"

        SPEAKER_LIST="/groups/wiz94156/Privacy-issues/MIA/seed-${SEED}/${MODEL}-unseen-speaker-similarity.csv"

        python train_speaker_level_model.py \
        --base_path $BASE_PATH \
        --output_path $OUTPUT_PATH \
        --model $MODEL \
        --num_workers 4 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --auxiliary_data_choice_size 1 \
        --n_epochs 20 \
        --speaker_list $SPEAKER_LIST


    done

done