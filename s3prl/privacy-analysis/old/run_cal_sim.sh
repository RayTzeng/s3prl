echo "[running wav2vec2...]"
python privacy-analysis/cal_cosine_similarity.py --base_path "/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output" --model "wav2vec2"

echo "[running cpc...]"
python privacy-analysis/cal_cosine_similarity.py --base_path "/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output" --model "modified_cpc"

echo "[running tera...]"
python privacy-analysis/cal_cosine_similarity.py --base_path "/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output" --model "tera"

echo "[running apc...]"
python privacy-analysis/cal_cosine_similarity.py --base_path "/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output" --model "apc"

echo "[running mockingjay...]"
python privacy-analysis/cal_cosine_similarity.py --base_path "/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output" --model "mockingjay"

echo "[running wav2vec...]"
python privacy-analysis/cal_cosine_similarity.py --base_path "/home/raytz/Disk/Privacy-Issues-Speech-BERT/Output" --model "wav2vec"