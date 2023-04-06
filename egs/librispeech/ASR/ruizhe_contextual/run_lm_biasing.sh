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

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 1: Ccontextualized ASR with WFST on-the-fly shallow fusion"

    for m in modified_beam_search ; do
        for epoch in $epochs; do
            for avg in $avgs; do
                # python -m pdb -c continue
                ./pruned_transducer_stateless7_context/decode.py \
                    --epoch $epoch \
                    --avg $avg \
                    --use-averaged-model $use_averaged_model \
                    --exp-dir $exp_dir \
                    --feedforward-dims  "1024,1024,2048,2048,1024" \
                    --max-duration 600 \
                    --decoding-method $m \
                    --context-dir "data/fbai-speech/is21_deep_bias/" \
                    --n-distractors $n_distractors \
                    --keep-ratio 1.0 --is-predefined true --no-wfst-lm-biasing false --biased-lm-scale 9 --no-encoder-biasing true --no-decoder-biasing true
                # --is-full-context true
                # --n-distractors 0
                # --no-encoder-biasing true --no-decoder-biasing true
                # --is-predefined true
                # --is-pretrained-context-encoder true
                # --no-wfst-lm-biasing false --biased-lm-scale 10
            done
        done
    done
fi

# /export/fs04/a12/rhuang/icefall_align/egs/spgispeech/ASR/pruned_transducer_stateless2/decode_pretrained.py
# https://github.com/huangruizhe/icefall/blob/biasing/egs/spgispeech/ASR/pruned_transducer_stateless2/decode_pretrained.py
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: contextualized ASR with WFST on-the-fly shallow fusion + RNNLM + LODR"

    path_to_pretrained_nnlm="/export/fs04/a12/rhuang/deep_smoothing/data_librispeech/icefall-librispeech-rnn-lm/"
    # ln -s $path_to_pretrained_nnlm/exp/pretrained.pt $path_to_pretrained_nnlm/exp/epoch-999.pt
    lm_type="rnn"

    tokens_ngram_order=2
    for m in modified_beam_search_LODR ; do
        for epoch in $epochs; do
            for avg in $avgs; do
                ./pruned_transducer_stateless7_context/decode.py \
                    --epoch $epoch \
                    --avg $avg \
                    --use-averaged-model $use_averaged_model \
                    --exp-dir $exp_dir \
                    --lang-dir data/lang_bpe_500 \
                    --max-duration 600 \
                    --decoding-method $m \
                    --beam-size 4 \
                    --max-contexts 4 \
                    --use-shallow-fusion true \
                    --lm-type $lm_type \
                    --lm-exp-dir $path_to_pretrained_nnlm/exp \
                    --lm-epoch 999 \
                    --lm-scale 0.4 \
                    --lm-avg 1 \
                    --rnn-lm-num-layers 3 \
                    --rnn-lm-tie-weights 1 \
                    --tokens-ngram $tokens_ngram_order \
                    --ngram-lm-scale -0.16 \
                    --context-dir "data/fbai-speech/is21_deep_bias/" \
                    --n-distractors $n_distractors \
                    --keep-ratio 1.0 --no-wfst-lm-biasing false --biased-lm-scale 11
                # --is-predefined true
                # --no-encoder-biasing true --no-decoder-biasing true
                # --no-wfst-lm-biasing false --biased-lm-scale 11
            done
        done
    done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Stage 3: compute entity-aware WER"
fi