#!/usr/bin/env bash
#$ -wd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
#$ -V
#$ -N decode
#$ -j y -o ruizhe_contextual/log/$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l ram_free=16G,mem_free=16G,gpu=1,hostname=!b*&!c18*&!c04*
#$ -q g.q

# &!octopod*

#### Activate dev environments and call programs
# mamba activate /home/rhuang/mambaforge/envs/efrat
mamba activate /home/rhuang/mambaforge/envs/efrat2
export PYTHONPATH=/export/fs04/a12/rhuang/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/export/fs04/a12/rhuang/k2/build/temp.linux-x86_64-cpython-38/lib/:$PYTHONPATH # for `import _k2`
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH

echo "python: `which python`"

#### Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu 1

# ngpus=4 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
# free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid
# [ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
# echo "Unable to get $ngpus GPUs"
# [ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
# [ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
#  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;
# export CUDA_VISIBLE_DEVICES="$free_gpu"
echo $CUDA_VISIBLE_DEVICES

#### Test running qsub
hostname
python3 -c "import torch; print(torch.__version__)"
nvidia-smi

####################################
# modified_beam_search
####################################

n_distractors=100
exp_dir=pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2_6k
exp_dir=pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2
context_suffix=""

epochs=99
avgs=1
use_averaged_model=$([ "$avgs" = 1 ] && echo "false" || echo "true")

stage=3
stop_stage=3

# download model from coe:
# cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
# mkdir -p $exp_dir
# scp -r rhuang@test1.hltcoe.jhu.edu:/exp/rhuang/icefall_latest/egs/spgispeech/ASR/$exp_dir/epoch-3.pt \
#   /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/${exp_dir}/epoch-99.pt

# No biasing at all
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  # greedy_search fast_beam_search
  # python -m pdb -c continue
  ./pruned_transducer_stateless2/decode.py \
      --epoch 999 \
      --avg 1 \
      --exp-dir "/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp/" \
      --bpe-model "/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model" \
      --max-duration 400 \
      --decoding-method "modified_beam_search" \
      --beam-size 4
fi

# Results:
# [using bad cuts] /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp/modified_beam_search/log-decode-epoch-999-avg-1-modified_beam_search-beam-size-4-2023-04-08-13-31-50
# beam_size=4:  10.84/6.88  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616646.out
# beam_size=20: 10.81/6.86  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616651.out

# No biasing at all + RNNLM + LODR
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  icefall_align_path="/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/"
  # rnnlm_dir="/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/LM/my-rnnlm-exp/"
  rnnlm_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-rnnlm-exp-1024-3-tied"
  lm_type=rnn
  lang_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-ngram-exp/bpe"
  rnn_lm_scale=0.2
  ngram_lm_scale=-0.1
  tokens_ngram_order=2
  python pruned_transducer_stateless2/decode.py \
      --epoch 999 \
      --avg 1 \
      --exp-dir "/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp/" \
      --bpe-model "/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model" \
      --max-duration 300 \
      --lang-dir $lang_dir \
      --decoding-method modified_beam_search_LODR \
      --beam 4 \
      --max-contexts 4 \
      --use-shallow-fusion true \
      --lm-type $lm_type \
      --lm-exp-dir $rnnlm_dir \
      --lm-epoch 7 \
      --lm-avg 5 \
      --lm-scale $rnn_lm_scale \
      --rnn-lm-embedding-dim 1024 \
      --rnn-lm-hidden-dim 1024 \
      --rnn-lm-num-layers 3 \
      --rnn-lm-tie-weights true \
      --tokens-ngram $tokens_ngram_order \
      --ngram-lm-scale $ngram_lm_scale
fi

# Results:
# 3616647 0.4-0.16: 10.68/7.01 /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616647.out
# 3616657 0.5-0.16: 11.05/7.41
# 3616658 0.3-0.16: 10.55/6.84 /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616658.out
# 3616660 0.3-0.1:  10.64/6.94
# 3616661 0.4-0.1:  10.95/7.3
# 3616662 0.4-0.2:  10.61/6.91
# 3616663 0.5-0.2:  10.88/7.22
# *3616664 0.2-0.1:  10.55/6.8  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616664.out 
# 3616665 0.4-0.0:  11.66/8.04
# 3616666 0.2-0.0:  10.76/7.02
# 3616667 0.3-0.0:  11.08/7.41
# 3616829 0.1-0.0:  10.68/6.86

# Use biasing
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  for m in modified_beam_search ; do
    for epoch in $epochs; do
      for avg in $avgs; do
        # python -m pdb -c continue
        ./pruned_transducer_stateless2_context/decode.py \
            --epoch $epoch \
            --avg $avg \
            --exp-dir $exp_dir \
            --bpe-model "/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model" \
            --max-duration 400 \
            --decoding-method $m \
            --beam-size 4 \
            --context-dir "data/rare_words" \
            --n-distractors $n_distractors \
            --keep-ratio 1.0 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 7 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
        # --context-dir "data/rare_words"
        # --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
        # --is-full-context true
        # --n-distractors 0
        # --no-encoder-biasing true --no-decoder-biasing true
        # --is-predefined true
        # --is-pretrained-context-encoder true
        # --no-wfst-lm-biasing false --biased-lm-scale 9
        # --is-predefined true --no-wfst-lm-biasing false --biased-lm-scale 9 --no-encoder-biasing true --no-decoder-biasing true
        #
        # lm-biasing (cheating+distractors): --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 11 --n-distractors 100
        # lm-biasing (slides): --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 3 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
        #
        # neural-biasing (slides): --max-duration 10 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
      done
    done
  done
fi

# Results (cuts100):
# cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
# ----------------
# rare_words_3k + 100 distractors
# no-biasing:           8.88/5.12 pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2_6k/modified_beam_search/log-decode-epoch-99-avg-1-modified_beam_search-beam-size-4-2023-04-08-22-28-10
# --biased-lm-scale 5:  8.49/4.99 /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616728.out
# --biased-lm-scale 6:  8.49/5.01 /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616723.out
# --biased-lm-scale 7:  8.41/5.0  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616724.out
# --biased-lm-scale 8:  8.33/4.94 /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616725.out
# --biased-lm-scale 9:  8.41/4.99 pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2_6k/modified_beam_search/log-decode-epoch-99-avg-1-modified_beam_search-beam-size-4-2023-04-08-22-33-53
# --biased-lm-scale 10: 8.25/4.96 /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616726.out
# --biased-lm-scale 11: 8.22/4.96 /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616727.out
# --biased-lm-scale 12: 8.88/5.55 3616730
# --biased-lm-scale 13: 10.02/6.9 3616731
# --biased-lm-scale 14: 10.26/7.11 3616732
# --biased-lm-scale 15: 3616733
# --biased-lm-scale 16: 3616734
# --biased-lm-scale 17: 3616735
# --biased-lm-scale 18: 3616736
# --biased-lm-scale 19: 3616737
# --biased-lm-scale 20: 13.01/9.93 3616738
# ----------------
# slides (cuts100):
# [20220108] /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20220108_slides/modified_beam_search_rnnlm_shallow_fusion_biased/log-decode-epoch-30-avg-15-modified_beam_search_rnnlm_shallow_fusion_biased-beam-size-10-3-ngram-lm-scale-0.01-rnnlm-lm-scale-0.1-biased-lm-scale-9.0-use-averaged-model-2023-01-09-07-24-39
# --biased-lm-scale 5:  8.77/5.15  pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2_6k/modified_beam_search/log-decode-epoch-99-avg-1-modified_beam_search-beam-size-4-2023-04-08-23-17-57
# --biased-lm-scale 9:  10.61/6.78 pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2_6k/modified_beam_search/log-decode-epoch-99-avg-1-modified_beam_search-beam-size-4-2023-04-08-23-06-07
# ----------------
# slides (cuts100, removed common words --- this can reduces the biaisng list size, see /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616785.out ):
# 0.1 3616786 8.88/5.12
# 1 3616785 8.81/5.03
# 2 3616778 8.69/5.06 
# 3 3616779 8.53/4.98
# 4 3616780 8.57/4.99
# 5 3616781 8.57/4.97
# 6 3616782 8.49/4.95
# 7 3616783 9.0/5.3
# 8 3616784 9.08/5.33
# ----------------
# slides (removed common words, cuts800)
# 0 3616794 10.22/6.5
# 3 3616788 10.13/6.46
# 4 3616789 10.13/6.45
# 5 3616790 10.15/6.46
# 6 3616791 10.14/6.47
# 7 3616792 10.22/6.51
# 8 3616793 10.21/6.52
# 9 3616795 10.28/6.62
# ----------------
# slides (removed common words, cuts_all)
# 3 3616830  10.77/6.86
# 4 3616831  10.77/6.86 10.63_48410     9.70_40179      64.86_2219      58.73_249       34.08_1246      14.59_298
# 5 3616832  10.77/6.87
# 6 3616847  10.78/6.88  ----?
# 7 3616848              ----?
# 8 3616841  10.88/6.97 10.73_48872     9.82_40676      63.78_2182      57.55_244       32.74_1197      14.59_298
# 9 3616843  10.96/7.05 10.81_49225     9.91_41037      63.64_2177      57.31_243       32.47_1187      14.64_299
# 10 3616842 11.07/7.16 10.92_49711     10.02_41504     63.43_2170      56.84_241       32.47_1187      14.73_301

# Neural biasing
# slides (removed common words, cuts100)
# beam_size_4     8.65    5.04    best for ec53


8.89_227        7.49_172        60.87_14        66.67_2 48.00_12        11.11_2
8.62_220        7.28_167        47.83_11        66.67_2 48.00_12        16.67_3

####################################
# LODR
####################################

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  rnnlm_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-rnnlm-exp-1024-3-tied"
  lm_type="rnn"

  lang_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-ngram-exp/bpe"
  rnn_lm_scale=0.2
  ngram_lm_scale=-0.1
  tokens_ngram_order=2
  for m in modified_beam_search_LODR ; do
    for epoch in $epochs; do
      for avg in $avgs; do
        ./pruned_transducer_stateless2_context/decode.py \
            --epoch $epoch \
            --avg $avg \
            --exp-dir $exp_dir \
            --bpe-model "/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model" \
            --max-duration 300 \
            --lang-dir $lang_dir \
            --decoding-method $m \
            --beam-size 4 \
            --max-contexts 4 \
            --use-shallow-fusion true \
            --lm-type $lm_type \
            --lm-exp-dir $rnnlm_dir \
            --lm-epoch 7 \
            --lm-avg 5 \
            --lm-scale $rnn_lm_scale \
            --rnn-lm-embedding-dim 1024 \
            --rnn-lm-hidden-dim 1024 \
            --rnn-lm-num-layers 3 \
            --rnn-lm-tie-weights true \
            --tokens-ngram $tokens_ngram_order \
            --ngram-lm-scale $ngram_lm_scale \
            --context-dir "data/rare_words" \
            --n-distractors $n_distractors \
            --keep-ratio 1.0 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 5 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
        # --is-predefined true
        # --no-encoder-biasing true --no-decoder-biasing true
        # --no-wfst-lm-biasing false --biased-lm-scale 10
        #
        # all: --is-predefined true --n-distractors 500 --no-wfst-lm-biasing false --biased-lm-scale 11
        # lm-biasing (slides) + lodr: --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 3 --slides "/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context${context_suffix}" --is-predefined true
      done
    done
  done
fi

# biased_lm_scale 4  3616834 10.5/6.79  10.39_47294     9.46_39173      63.81_2183      58.25_247       32.88_1202      14.29_292
# biased_lm_scale 5  3616849
# biased_lm_scale 6  3616835 10.52/6.82 10.41_47400     9.50_39320      63.46_2171      56.60_240       32.19_1177      14.19_290  # /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3616835.out
# biased_lm_scale 7  3616839 10.57/6.86 10.46_47610     9.54_39511      63.49_2172      56.60_240       32.17_1176      14.34_293
# biased_lm_scale 8  3616840 10.63/6.92 10.52_47891     9.61_39795      63.26_2164      56.60_240       31.87_1165      14.49_296
# biased_lm_scale 9  3616836 10.7/6.98  10.59_48205     9.68_40085      62.85_2150      56.13_238       31.89_1166      14.73_301
# biased_lm_scale 10 3616837 10.78/7.06 10.67_48563     9.76_40420      62.61_2142      56.37_239       31.87_1165      14.73_301


####################################
# Export averaged models 
# https://icefall.readthedocs.io/en/latest/model-export/export-model-state-dict.html#how-to-export
####################################

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  python pruned_transducer_stateless7_context/export.py \
    --exp-dir $exp_dir \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch $epochs \
    --avg $avgs

  mv $exp_dir/pretrained.pt $exp_dir/stage1.pt
fi
