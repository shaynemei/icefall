# /export/fs04/a12/rhuang/contextualizedASR/spgi/run.sh

cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
mkdir -p data/rare_words

train_text="/export/fs04/a12/rhuang/contextualizedASR/data/lm/spgi.train.norm.txt"
time cat $train_text | cut -d" " -f2- | \
    tr ' ' '\n' | sort | uniq -c | sort -nr | awk '{print $2" "$1}' > data/rare_words/all_words.count.txt
# | awk '{print $2" "$1}' | sort -k1,1

topk=6000
topk=3000

cat data/rare_words/all_words.count.txt | head -n $topk | \
  awk '{print $1}' > data/rare_words/common_words_3k.txt

tail +$( expr $topk + 1 ) data/rare_words/all_words.count.txt | \
  awk '{print $1}' > data/rare_words/all_rare_words.txt

wc data/rare_words/*.txt

########################
# Rare words stats
########################

spgi_word_count="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/rare_words/all_words.count.txt"
cat $spgi_word_count | head -n $topk | \
  awk '{sum+=$2;}END{print sum;}'

tail +$( expr $topk + 1 ) $spgi_word_count | \
  awk '{sum+=$2;}END{print sum;}'

echo "" | awk '{x=1314950/43966593}END{print x;}'

# 6k: 1314950/43966593=0.0299079
# 3k: 2831893/42449650=0.0667118103

libri_word_count="/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/data/fbai-speech/is21_deep_bias/words/all_words.count.txt"
cat $libri_word_count | head -n5016 | \
  awk '{sum+=$2;}END{print sum;}'

tail +5017 $libri_word_count | \
  awk '{sum+=$2;}END{print sum;}'

echo "" | awk '{x=946854/8455308}END{print x;}'
# 0.111983

# 5k: 946854/8455308=0.111983

