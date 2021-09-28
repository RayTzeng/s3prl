BASE_PATH=/mnt/storage1/Speech/VCTK-Corpus/wav48
OUTPUT_PATH=/mnt/storage2/speech_data/VCTK-Feature
MODEL_PATH=/mnt/ship_groups/public/ssm_pretrain/TERA
OUTPUT_PREFIX=mel20ms
CONFIG=/livingrooms/wiz94156/Privacy-issues-speech-bert/s3prl/s3prl/upstream/baseline/mel20ms.yaml
for MODEL in tera_100hr
do
echo "running ${MODEL}"

OUTPUT_PREFIX=$MODEL
python preprocess_feature_VCTK.py --base_path $BASE_PATH \
    --output_path $OUTPUT_PATH \
    --model $MODEL \
    --output_prefix $OUTPUT_PREFIX 
    #--state_dict $MODEL_PATH/$MODEL/states-epoch-224.ckpt \
    #--model_cfg $MODEL_PATH/$MODEL/config_model.yaml \

    #--config $CONFIG
done
