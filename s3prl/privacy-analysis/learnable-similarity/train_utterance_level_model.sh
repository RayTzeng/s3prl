BASE_PATH=/livingrooms/wiz94156/LibriSpeech-Feature/
OUTPUT_PATH=/groups/wiz94156/Privacy-issues/MIA/100uttr-learnable/similarity-model

#  hubert_large_ll60k hubert_xtralarge_ll60k wav2vec2_large_960 wav2vec2_large_ll60k tera_100hr
for SEED in 57 666 1450
do
    OUTPUT_PATH="/groups/wiz94156/Privacy-issues/MIA/fake-500uttr-learnable/similarity-model-seed-${SEED}"
    
    # hubert wav2vec2 modified_cpc tera 
    for MODEL in wav2vec2_large_ll60k
    do
        echo "[running seed ${SEED}]"
        echo "[training utterance-level similarity model for ${MODEL}...]"

        UTTERANCE_LIST="/groups/wiz94156/Privacy-issues/MIA/seed-${SEED}/${MODEL}-unseen-utterance-similarity.csv"

        python train_utterance_level_model.py \
        --base_path $BASE_PATH \
        --output_path $OUTPUT_PATH \
        --model $MODEL \
        --num_workers 4 \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --n_epochs 20 \
        --auxiliary_data_choice_size 500 \
        --lr 1e-5 \
        --utterance_list $UTTERANCE_LIST \
        --seed $SEED


    done

done
