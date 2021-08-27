BASE_PATH=/home/raytz/Disk/LibriSpeech
OUTPUT_PATH=/home/raytz/Disk/LibriSpeech-Feature

for MODEL in wav2vec2_large_960 wav2vec2_large_ll60k tera_100hr 
do
echo "running ${MODEL}"

for SPLIT in train-clean-100 test-clean test-other dev-clean dev-other
do
echo "[running ${SPLIT}]"
python preprocess_feature_LibriSpeech.py --base_path $BASE_PATH \
    --split $SPLIT \
    --output_path $OUTPUT_PATH \
    --model $MODEL
done

done