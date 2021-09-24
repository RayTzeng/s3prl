BASE_PATH=/livingrooms/wiz94156/LibriSpeech-Feature/
for SEED in 57 666 1450
do

OUTPUT_PATH="/groups/wiz94156/Privacy-issues/MIA/seed-${SEED}"
for code in compute-speaker-similarity.py 
do
#  
for MODEL in hubert wav2vec2 modified_cpc tera
do
	echo "running seed-${SEED}"
	echo "running ${code}"
	echo "running ${MODEL}"
	python $code --base_path $BASE_PATH --output_path $OUTPUT_PATH --model $MODEL --seed $SEED

done

done

done
