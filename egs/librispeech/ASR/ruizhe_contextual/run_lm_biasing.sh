#!/usr/bin/env bash

# /export/fs04/a12/rhuang/icefall/
# branch: biasing
#
# For EC53:
# https://github.com/huangruizhe/icefall/blob/biasing/egs/spgispeech/ASR/pruned_transducer_stateless7/run_contextualized.sh

stage=-1
stop_stage=100

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# dependencies:
# conda install pdftotext

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Stage 0: Preparing a list of biasing phrase"

    if [ ! -d "data/fbai-speech/is21_deep_bias/" ]; then
        mkdir -p data/fbai-speech/
        git clone git@github.com:facebookresearch/fbai-speech.git data/fbai-speech
    fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Stage 1: Convert biasing list to WFST representation (optional)"

    # python local/context/prepare_context_graph1.py \
    #     --bpe-model-file "/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/data/lang_bpe_500/unigram_500.model" \
    #     --backoff-id 500 \
    #     --context-dir "data/lang/context" \
fi

# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/pruned_transducer_stateless2/decode_pretrained.py
# https://github.com/huangruizhe/icefall/blob/biasing/egs/spgispeech/ASR/pruned_transducer_stateless2/decode_pretrained.py
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: contextualized ASR with WFST on-the-fly shallow fusion"

    rnnlm_dir="/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/LM/my-rnnlm-exp"
    lang_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-ngram-exp/mix"

    python pruned_transducer_stateless2/decode_pretrained.py \
        --checkpoint tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp/pretrained.pt \
        --bpe-model tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/data/lang_bpe_500/bpe.model \
        --exp-dir tmp/icefall-asr-spgispeech-pruned-transducer-stateless2/exp20211206 \
        --decoding-method modified_beam_search_rnnlm_shallow_fusion_biased \
        --beam-size 4 \
        --rnn-lm-scale 0.3 \
        --rnn-lm-exp-dir $rnnlm_dir \
        --rnn-lm-epoch 13 \
        --rnn-lm-avg 1 \
        --rnn-lm-num-layers 2 \
        --rnn-lm-hidden-dim 200 \
        --rnn-lm-embedding-dim 800 \
        --rnn-lm-tie-weights false \
        --lang-dir $lang_dir \
        --biased-lm-scale 10
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: compute entity-aware WER"
fi