#!/usr/bin/env bash

pip install nltk
pip install pytorch
pip install pytorch_pretrained_bert
python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/share/nltk_data')"

python3 -m qanta.tfidf web
