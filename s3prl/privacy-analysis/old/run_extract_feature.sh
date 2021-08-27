MODEL=tera

DATASET=LibriSpeech/test-clean
AUDIO_DIR=/home/raytz/Disk/Privacy-Issues-Speech-BERT/Audio/${DATASET}
OUTPUT_DIR=/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output/${MODEL}/${DATASET}


echo $AUDIO_DIR
echo $OUTPUT_DIR

python privacy-analysis/extract_feature.py --audio_dir $AUDIO_DIR \
    --feature_output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --num_workers 4 \
    --model $MODEL \
    --file_type "flac"

DATASET=LibriSpeech/train-clean-100
AUDIO_DIR=/home/raytz/Disk/Privacy-Issues-Speech-BERT/Audio/${DATASET}
OUTPUT_DIR=/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output/${MODEL}/${DATASET}


echo $AUDIO_DIR
echo $OUTPUT_DIR

python privacy-analysis/extract_feature.py --audio_dir $AUDIO_DIR \
    --feature_output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --num_workers 4 \
    --model $MODEL \
    --file_type "flac"

DATASET=GoogleTTS-LibriSpeech/train-clean-100
AUDIO_DIR=/home/raytz/Disk/Privacy-Issues-Speech-BERT/Audio/${DATASET}
OUTPUT_DIR=/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output/${MODEL}/${DATASET}


echo $AUDIO_DIR
echo $OUTPUT_DIR

python privacy-analysis/extract_feature.py --audio_dir $AUDIO_DIR \
    --feature_output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --num_workers 4 \
    --model $MODEL \

DATASET=CMU18/bdl-clb-rms-slt/
AUDIO_DIR=/home/raytz/Disk/Privacy-Issues-Speech-BERT/Audio/${DATASET}
OUTPUT_DIR=/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output/${MODEL}/${DATASET}


echo $AUDIO_DIR
echo $OUTPUT_DIR

python privacy-analysis/extract_feature.py --audio_dir $AUDIO_DIR \
    --feature_output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --num_workers 4 \
    --model $MODEL \
