#!/usr/bin/env bash

# /export/fs04/a12/rhuang/icefall/
# branch: biasing
#
# For EC53:
# https://github.com/huangruizhe/icefall/blob/biasing/egs/spgispeech/ASR/pruned_transducer_stateless7/run_contextualized.sh

epoch=1564000
avg=1
exp_dir=/seasalt-t4gpu-02/chmei/mammoth/examples/zh_tw/stt_data_validation/exp_orig/icefall
decoding_method="modified_beam_search"
path_cuts="/seasalt-t4gpu-02/chmei/mammoth/icefall/egs/seasalt_zh_project20k_cdsw/ASR/data/icefall_rnnlm_tiny_val/manifests/cuts_icefall_rnnlm_tiny_val.jsonl.gz"
n_distractors=5

./pruned_transducer_stateless7_context/decode_seasalt.py \
    --epoch $epoch \
    --avg $avg \
    --use-averaged-model False \
    --exp-dir $exp_dir \
    --num-encoder-layers 18 \
    --dim-feedforward 2048 \
    --nhead 8 \
    --encoder-dim 512 \
    --decoder-dim 512 \
    --joiner-dim 512 \
    --max-duration 600 \
    --feedforward-dims  "1024,1024,2048,2048,1024" \
    --decoding-method $decoding_method \
    --beam-size 4 \
    --context-dir "data/lion-travel" \
    --n-distractors $n_distractors \
    --keep-ratio 1.0 \
    --is-predefined false \
    --no-wfst-lm-biasing false \
    --biased-lm-scale 9 \
    --no-encoder-biasing true \
    --no-decoder-biasing true \
    --lang-dir /seasalt-t4gpu-02/chmei/mammoth/icefall/egs/seasalt_zh_project20k_cdsw/ASR/data/lang_char \
    --cuts "${path_cuts}"

#log "Stage 3: contextualized ASR with WFST on-the-fly shallow fusion + RNNLM + LODR"
#
#path_to_pretrained_nnlm="/seasalt-t4gpu-02/chmei/mammoth/icefall/egs/seasalt_zh_project20k_cdsw/ASR/rnnlm_char/test"
## ln -s $path_to_pretrained_nnlm/exp/pretrained.pt $path_to_pretrained_nnlm/exp/epoch-99.pt
#lm_type="rnn"
#
#tokens_ngram_order=2
#./pruned_transducer_stateless7_context/decode.py \
#    --epoch $epoch \
#    --avg $avg \
#    --use-averaged-model $use_averaged_model \
#    --exp-dir $exp_dir \
#    --lang-dir data/lang_bpe_500 \
#    --max-duration 600 \
#    --decoding-method $m \
#    --beam-size 4 \
#    --max-contexts 4 \
#    --use-shallow-fusion true \
#    --lm-type $lm_type \
#    --lm-exp-dir $path_to_pretrained_nnlm/exp \
#    --lm-epoch 99 \
#    --lm-scale 0.4 \
#    --lm-avg 1 \
#    --rnn-lm-num-layers 3 \
#    --rnn-lm-tie-weights 1 \
#    --tokens-ngram $tokens_ngram_order \
#    --ngram-lm-scale -0.16 \
#    --context-dir "data/fbai-speech/is21_deep_bias/" \
#    --n-distractors $n_distractors \
#    --keep-ratio 1.0 --no-wfst-lm-biasing false --biased-lm-scale 11