# modified from https://github.com/dbamman/latin-bert/blob/cd6bea9f7ff84ff4b18c172f3d5719d1d3198e69/scripts/download.sh
mkdir -p latin-bert/models
cd latin-bert/models
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Te_14UB-DZ8wYPhHGyDg7LadDTjNzpti' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Te_14UB-DZ8wYPhHGyDg7LadDTjNzpti" -O latin_bert.tar && rm -f /tmp/cookies.txt
tar -xf latin_bert.tar
rm latin_bert.tar
mkdir subword_tokenizer_latin
cd subword_tokenizer_latin
wget https://github.com/dbamman/latin-bert/raw/master/models/subword_tokenizer_latin/latin.subword.encoder
cd ../../..

# Latin SBERT
wget "https://www.dropbox.com/s/o7fo95wbgl4tk8e/best_model.tar?dl=1" -O best_model.tar
tar -xf best_model.tar
rm best_model.tar

# also download .tess files
cd data
wget https://github.com/tesserae/tesserae/raw/master/texts/la/vergil.aeneid.tess
wget https://github.com/tesserae/tesserae/raw/master/texts/la/lucan.bellum_civile/lucan.bellum_civile.part.1.tess

# as well as Tesserae Latin lexicon
wget https://raw.githubusercontent.com/tesserae/tesserae/master/data/common/la.lexicon.csv

# also aurelberra's Latin stopwords list
wget https://github.com/aurelberra/stopwords/raw/master/stopwords_latin.json
cd ..
