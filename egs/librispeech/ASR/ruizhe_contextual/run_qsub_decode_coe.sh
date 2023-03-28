#!/usr/bin/env bash
#$ -wd /exp/rhuang/icefall_latest/egs/librispeech/ASR
#$ -V
#$ -N decode
#$ -j y -o ruizhe_contextual/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=1
#$ -q gpu.q@@v100

# #$ -q gpu.q@@v100
# #$ -q gpu.q@@rtx

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
hostname
nvidia-smi

export PATH="/exp/rhuang/mambaforge/envs/icefall2/bin/":$PATH
which python
export PATH="/exp/rhuang/gcc-7.2.0/mybuild/bin":$PATH
export PATH=/exp/rhuang/cuda-10.2/bin/:$PATH
which nvcc
export LD_LIBRARY_PATH=/exp/rhuang/cuda-10.2/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/exp/rhuang/gcc-7.2.0/mybuild/lib64:$LD_LIBRARY_PATH

# k2
K2_ROOT=/exp/rhuang/k2/
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$K2_ROOT/build_debug_cuda10.2/lib:$PYTHONPATH # for `import _k2`

# kaldifeat
export PYTHONPATH=/exp/rhuang/kaldifeat/build/lib:/exp/rhuang/kaldifeat/kaldifeat/python:$PYTHONPATH

# icefall
# export PYTHONPATH=/exp/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/exp/rhuang/icefall_latest/:$PYTHONPATH
# export PYTHONPATH=/exp/rhuang/icefall/icefall/transformer_lm/:$PYTHONPATH

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"

# greedy_search fast_beam_search
for m in modified_beam_search ; do
  for epoch in 26; do
    for avg in 10; do
      ./pruned_transducer_stateless7_context/decode.py \
          --epoch $epoch \
          --avg $avg \
          --use-averaged-model true \
          --exp-dir ./pruned_transducer_stateless7_context/exp/exp_libri_full \
          --feedforward-dims  "1024,1024,2048,2048,1024" \
          --max-duration 600 \
          --decoding-method $m \
          --context-dir "data/fbai-speech/is21_deep_bias/" \
          --context-n-words 100 \
          --keep-ratio 1.0
    done
  done
done

# On CLSP grid:
# cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
# for m in modified_beam_search ; do   for epoch in 10; do     for avg in 1; do       ./pruned_transducer_stateless7_context/decode.py           --epoch $epoch           --avg $avg           --use-averaged-model 0           --exp-dir ./pruned_transducer_stateless7_context/exp/exp_libri_full           --feedforward-dims  "1024,1024,2048,2048,1024"           --max-duration 400           --decoding-method $m;     done;   done; done
# /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full/modified_beam_search/log-decode-epoch-10-avg-1-modified_beam_search-beam-size-4-2023-03-27-17-25-08


for m in modified_beam_search ; do   for epoch in 24; do     for avg in 5; do       ./pruned_transducer_stateless7_context/decode.py           --epoch $epoch           --avg $avg           --use-averaged-model 0           --exp-dir ./pruned_transducer_stateless7_context/exp/exp_libri_full           --feedforward-dims  "1024,1024,2048,2048,1024"           --max-duration 400           --decoding-method $m;     done;   done; done