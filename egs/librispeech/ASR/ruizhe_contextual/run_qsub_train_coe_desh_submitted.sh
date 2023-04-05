#!/usr/bin/env bash
#$ -wd /exp/draj/mini_scale_2022/icefall/egs/librispeech/ASR 
#$ -V
#$ -N train
#$ -j y -o pruned_transducer_stateless7_context/exp/log/log-$JOB_NAME-$JOB_ID.out
#$ -M draj2@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=4
#$ -q gpu.q@@rtx

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

max_duration=700
n_distractors=-1
# max_duration=700
# n_distractors=500
# max_duration=100

full_libri=true
full_libri_name=$([ "$full_libri" = true ] && echo "full" || echo "100")

# Continue training from pretrained.pt
path_to_pretrained_asr_model=/exp/rhuang/librispeech/pretrained2/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_stage1
mkdir -p $exp_dir
if [ ! -f $exp_dir/epoch-1.pt ]; then
  ln -s /exp/rhuang/icefall_latest/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/epoch-10.pt $exp_dir/epoch-1.pt
fi

ray_icefall_base=/exp/rhuang/icefall_latest/egs/librispeech/ASR/
python ${ray_icefall_base}/pruned_transducer_stateless7_context/train.py \
  --world-size 4 \
  --full-libri $full_libri \
  --use-fp16 true \
  --max-duration $max_duration \
  --exp-dir $exp_dir \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12535 \
  --num-workers 10 \
  --base-lr 0.1 \
  --context-dir "${ray_icefall_base}/data/fbai-speech/is21_deep_bias/" \
  --keep-ratio 1.0 \
  --start-epoch 2 \
  --num-epochs 30 \
  --n-distractors $n_distractors


