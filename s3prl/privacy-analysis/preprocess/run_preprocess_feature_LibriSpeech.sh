BASE_PATH=/groups/public/LibriSpeech
OUTPUT_PATH=/livingrooms/wiz94156/LibriSpeech-Feature
OUTPUT_PREFIX=mel20ms
CONFIG=/livingrooms/wiz94156/Privacy-issues-speech-bert/s3prl/s3prl/upstream/baseline/mel20ms.yaml


for MODEL in baseline_local
do
echo "running ${MODEL}"

for SPLIT in train-clean-100 test-clean test-other dev-clean dev-other
do
echo "[running ${SPLIT}]"
python preprocess_feature_LibriSpeech.py --base_path $BASE_PATH \
    --split $SPLIT \
    --output_path $OUTPUT_PATH \
    --model $MODEL \
    --output_prefix $OUTPUT_PREFIX \
    --config $CONFIG
done

done
