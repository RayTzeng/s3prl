# python speaker-level-MIA.py --base_path "/home/raytz/Disk/LibriSpeech-Feature/" --output_path "/home/raytz/Disk/" --model "hubert"
# python speaker-level-MIA.py --base_path "/home/raytz/Disk/LibriSpeech-Feature/" --output_path "/home/raytz/Disk/" --model "wav2vec2"
# python speaker-level-MIA.py --base_path "/home/raytz/Disk/LibriSpeech-Feature/" --output_path "/home/raytz/Disk/" --model "modified_cpc"
# python speaker-level-MIA.py --base_path "/home/raytz/Disk/LibriSpeech-Feature/" --output_path "/home/raytz/Disk/" --model "tera"

python speaker-level-MIA.py --base_path "/home/raytz/Disk/LibriSpeech-Feature/" --output_path "/home/raytz/Disk/" --model "tera_100hr"

python speaker-level-MIA.py --base_path "/home/raytz/Disk/LibriSpeech-Feature/" --output_path "/home/raytz/Disk/" --model "wav2vec2_large_960"

python speaker-level-MIA.py --base_path "/home/raytz/Disk/LibriSpeech-Feature/" --output_path "/home/raytz/Disk/" --model "wav2vec2_large_ll60k"
