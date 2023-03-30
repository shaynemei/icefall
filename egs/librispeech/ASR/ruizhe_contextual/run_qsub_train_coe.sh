#!/usr/bin/env bash
#$ -wd /exp/rhuang/icefall_latest/egs/librispeech/ASR
#$ -V
#$ -N train
#$ -j y -o ruizhe_contextual/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=4
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

# python pruned_transducer_stateless7/train.py \
#   --world-size 4 \
#   --num-epochs 30 \
#   --full-libri false \
#   --use-fp16 true \
#   --max-duration 750 \
#   --exp-dir pruned_transducer_stateless7/exp/exp_libri_100 \
#   --feedforward-dims  "1024,1024,2048,2048,1024" \
#   --master-port 12535

context_n_words=100
max_duration=700
# context_n_words=500
# max_duration=100

# path_to_pretrained_asr_model=/exp/rhuang/librispeech/pretrained2/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${context_n_words}
# mkdir -p $exp_dir
# if [ ! -f $exp_dir/epoch-1.pt ]; then
#   ln -s $path_to_pretrained_asr_model/exp/epoch-30.pt $exp_dir/epoch-1.pt
# fi

# continue training from the wrong model
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${context_n_words}_continue
mkdir -p $exp_dir
ln -s /exp/rhuang/icefall_latest/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_lowerwrong/epoch-30.pt \
  $exp_dir/epoch-1.pt

python pruned_transducer_stateless7_context/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --full-libri true \
  --use-fp16 true \
  --max-duration $max_duration \
  --exp-dir $exp_dir \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12535 \
  --num-workers 10 \
  --base-lr 0.1 \
  --context-dir "data/fbai-speech/is21_deep_bias/" \
  --keep-ratio 1.0 \
  --start-epoch 2 \
  --context-n-words $context_n_words

# --start-batch 

# context size 100 (wrong due to lowercased biasing list):
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10576003.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10577155.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10578161.out

# context size 500 (wrong due to lowercased biasing list):
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10579396.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10579398.out

# context size 100
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10579501.out

# context size 100 (trained from the wrong model)
# /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/log/train-3611947.out
# /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/log/train-3611979.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10579608.out