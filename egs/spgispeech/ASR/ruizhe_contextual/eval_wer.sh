cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR

# Look for the cuts/segmentatio that has good quality
grep 47026 /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp*/*/wer/scoring_kaldi/wer
# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_50.0/modified_beam_search_rnnlm_shallow_fusion_biased/wer/scoring_kaldi/wer:%WER 10.33 [ 47026 / 455331, 11073 ins, 8801 del, 27152 sub ]

less /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_50.0/modified_beam_search_rnnlm_shallow_fusion_biased/log-decode-epoch-30-avg-15-modified_beam_search_rnnlm_shallow_fusion_biased-beam-size-10-3-ngram-lm-scale-0.01-rnnlm-lm-scale-0.1-biased-lm-scale-9.0-use-averaged-model-2023-01-09-07-23-11
# /export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics11_temp/cuts_no_feat_20230109050816_merged.jsonl.gz

####################################
# prepare kaldi dir: text
####################################

# BAD!!
# cuts="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_sp_gentle/20220129/cuts3.jsonl.gz" 
# cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"

# GOOD
cuts="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics11_temp/cuts_no_feat_20230109050816_merged.jsonl.gz"
cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"

mkdir -p data/kaldi
python -c '''
from lhotse import CutSet
cuts = CutSet.from_file("/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics11_temp/cuts_no_feat_20230109050816_merged.jsonl.gz")
for cut in cuts:
  uid = cut.supervisions[0].id
  text = cut.supervisions[0].text
  print(f"{uid}\t{text}")
''' | tr -s " " | sort > data/kaldi/text
wc data/kaldi/text

cat data/kaldi/text | awk '{print $1" aaa"}' > data/kaldi/utt2spk

# Also refer to:
# /export/fs04/a12/rhuang/contextualizedASR/spgi/run.sh

####################################
# prepare ref
####################################

# Refer to:
# /export/fs04/a12/rhuang/contextualizedASR/lm/ngram.sh

# Can we reuse some ref files produced before?
grep "TSM_2020_Q1_20200416_00-56-37-680_00-57-41-850_461" /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp*/*/wer/ref.txt
# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20230124/modified_beam_search/wer/ref.txt:TSM_2020_Q1_20200416_00-56-37-680_00-57-41-850_461 and the 7 nanometer was pretty tight at the beginning of the year . charlie we cannot hear you clearly
# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20230129/modified_beam_search/wer/ref.txt:TSM_2020_Q1_20200416_00-56-37-680_00-57-41-850_461 and the 7 nanometer was pretty tight at the beginning of the year . charlie we cannot hear you clearly
# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_50.0/modified_beam_search_rnnlm_shallow_fusion_biased/wer/ref*
cp /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_50.0/modified_beam_search_rnnlm_shallow_fusion_biased/wer/ref* /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/ref/.

# normalize ref
# https://unix.stackexchange.com/questions/14838/sed-one-liner-to-delete-everything-between-a-pair-of-brackets
# https://stackoverflow.com/questions/27825977/using-sed-to-delete-a-string-between-parentheses
mkdir -p data/kaldi/ref/
mamba activate whisper
# /export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics/text
cat /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/text | \
    sed -e 's/\[[^][]*\]//g' | \
    python /export/fs04/a12/rhuang/contextualizedASR/earnings21/whisper_normalizer.py \
    --mode "kaldi_rm_sym" | sort \
> data/kaldi/ref/ref.txt

# skip entity tagging (or go to "ec53/prepare_kaldi_dir.sh" to see how to do tagging)
cp /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/ref/ref.* $wer_dir/.

# entity tagging (go to local/ner_spacy.sh and modify the paths there)
qsub /export/fs04/a12/rhuang/contextualizedASR/local/ner_spacy.sh
# /export/fs04/a12/rhuang/contextualizedASR/log-ner-3616648.out

wer_dir="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/ref/"
python local/ner_spacy.py \
  --text "$wer_dir/ref.txt" \
  --out "$wer_dir/ref.ner" \
  --entities "$wer_dir/ref.entities" \
  --raw "$wer_dir/ref.ner.raw"

####################################
# prepare context/biasing list
####################################



####################################
# eval wer
####################################

eval_wer () {
  icefall_hyp=$1

  # cuts="data/ec53_manifests/cuts_ec53_trimmed2.jsonl.gz"
  # kaldi_data="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics"
  # cuts="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_sp_gentle/cuts.jsonl.gz"
  # kaldi_data="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_sp_gentle/"
  cuts="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics11_temp/cuts_no_feat_20230109050816_merged.jsonl.gz" 
  cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"
  kaldi_data="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/"

  python /export/fs04/a12/rhuang/contextualizedASR/local/recogs_to_text.py \
    --input $icefall_hyp \
    --out ${icefall_hyp%.*}.text \
    --cuts $cuts

  wer_dir=$(dirname $icefall_hyp)/wer
  mkdir -p $wer_dir

  cat ${icefall_hyp%.*}.text | \
    python /export/fs04/a12/rhuang/contextualizedASR/earnings21/whisper_normalizer.py \
    --mode "kaldi_rm_sym" | sort \
  > $wer_dir/hyp.txt

  # cat $kaldi_data/text | \
  #   sed -e 's/\[[^][]*\]//g' | \
  #   python /export/fs04/a12/rhuang/contextualizedASR/earnings21/whisper_normalizer.py \
  #   --mode "kaldi_rm_sym" | sort \
  # > $wer_dir/ref.txt
  # qsub /export/fs04/a12/rhuang/contextualizedASR/local/ner_spacy.sh
  # or:
  # cp /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108/modified_beam_search/wer/ref.* $wer_dir/.
  cp /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/kaldi/ref/ref.* $wer_dir/.

  # compute WER
  ref=$(realpath $wer_dir/ref.txt)
  datadir=$(realpath ${kaldi_data})
  hyp=$(realpath $wer_dir/hyp.txt)
  decode=$(dirname $hyp)
  wc -l $ref $hyp

  cd /export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/
  # . /export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/path.sh
  bash /export/fs04/a12/rhuang/espnet/egs2/spgispeech/asr1/local/score_kaldi_light.sh \
    $ref $hyp $datadir $decode
  cd -

  # export PYTHONPATH=/export/fs04/a12/rhuang/contextualizedASR/:$PYTHONPATH
  python /export/fs04/a12/rhuang/contextualizedASR/local/check_ner2.py \
    --special_symbol "'***'" \
    --per_utt $decode/scoring_kaldi/wer_details/per_utt \
    --ref_ner $decode/ref.ner #--biasing_list "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context/"
}

mamba activate whisper
export PYTHONPATH=/export/fs04/a12/rhuang/contextualizedASR/:$PYTHONPATH

recogs=/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp/modified_beam_search/recogs-ec53-epoch-999-avg-1-modified_beam_search-beam-size-4.txt
eval_wer $recogs

# modified beam search (baseline)
# [20230108] /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108/modified_beam_search/log-decode-epoch-30-avg-15-modified_beam_search-beam-size-20-3-ngram-lm-scale-0.01-use-averaged-model-2023-01-09-05-11-29 
# /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616646.out

# modified beam search + rnnlm + lodr
# 3616647

# modified beam search + lm biasing
# 

