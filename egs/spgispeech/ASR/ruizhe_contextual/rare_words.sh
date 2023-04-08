# /export/fs04/a12/rhuang/contextualizedASR/spgi/run.sh

cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
mkdir -p data/rare_words

train_text="/export/fs04/a12/rhuang/contextualizedASR/data/lm/spgi.train.norm.txt"
time cat $train_text | cut -d" " -f2- | \
    tr ' ' '\n' | sort | uniq -c | sort -nr | awk '{print $2" "$1}' > data/rare_words/all_words.count.txt
# | awk '{print $2" "$1}' | sort -k1,1

cat data/rare_words/all_words.count.txt | head -n6000 | \
  awk '{print $1}' > data/rare_words/common_words_6k.txt

tail +6001 data/rare_words/all_words.count.txt | \
  awk '{print $1}' > data/rare_words/all_rare_words.txt

wc data/rare_words/*.txt