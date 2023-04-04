#!/usr/bin/env bash
#$ -wd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
#$ -V
#$ -N train
#$ -j y -o ruizhe_contextual/log/$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l ram_free=32G,mem_free=32G,gpu=8,hostname=!b*
#$ -q g.q

# -q 4gpu.q
# &!octopod*

world_size=8

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
# source /home/gqin2/scripts/acquire-gpu $world_size

ngpus=$world_size # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid
[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
 echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;
export CUDA_VISIBLE_DEVICES="$free_gpu"
echo $CUDA_VISIBLE_DEVICES

#### Test running qsub
hostname
python3 -c "import torch; print(torch.__version__)"
nvidia-smi

#### Your script
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# python3 tdnn_lstm_ctc/train.py --world-size 4
# python3 tdnn_lstm_ctc/train.py --world-size 4 --master-port 12355
# python3 tdnn_lstm_ctc2/train.py --world-size 4

# python3 tdnn_lstm_ctc/train_bpe.py --world-size 4 

# libri100
# python pruned_transducer_stateless7_context/train.py \
#   --world-size 1 \
#   --num-epochs 30 \
#   --full-libri false \
#   --use-fp16 true \
#   --max-duration 300 \
#   --exp-dir pruned_transducer_stateless7_context/exp/exp_libri_100 \
#   --feedforward-dims  "1024,1024,2048,2048,1024" \
#   --master-port 12535 \
#   --start-epoch 2

n_distractors=100
# n_distractors=-1
max_duration=120
# n_distractors=500
# max_duration=100

full_libri=true
full_libri_name=$([ "$full_libri" = true ] && echo "full" || echo "100")

# path_to_pretrained_asr_model=/exp/rhuang/librispeech/pretrained2/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}
# mkdir -p $exp_dir
# if [ ! -f $exp_dir/epoch-1.pt ]; then
#   ln -s $path_to_pretrained_asr_model/exp/epoch-30.pt $exp_dir/epoch-1.pt
# fi

# continue training from the wrong model
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_${full_libri_name}_c${n_distractors}_continue3
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_${full_libri_name}_c${n_distractors}_bert_stage1
mkdir -p $exp_dir
# ln -sf /export/fs04/a12/rhuang/deep_smoothing/data_librispeech/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/epoch-30.pt \
#   $exp_dir/epoch-1.pt
# ln -s /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_wronglower/epoch-30.pt \
#   $exp_dir/epoch-1.pt
ln -sf /export/fs04/a12/rhuang/deep_smoothing/data_librispeech/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/pretrained.pt \
  $exp_dir/epoch-1.pt

python pruned_transducer_stateless7_context/train.py \
  --world-size $world_size \
  --num-epochs 30 \
  --full-libri $full_libri \
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
  --n-distractors $n_distractors --is-pretrained-context-encoder true --n-distractors 0 --is-full-context true

# --is-pretrained-context-encoder true
# --n-distractors 0 --is-full-context true


