cd /exp/rhuang/icefall_latest/egs/spgispeech/ASR/
# export PATH=/export/fs04/a12/rhuang/git-lfs-3.2.0/:$PATH
git lfs install
git lfs version
mkdir -p pretrained
cd pretrained; git clone https://huggingface.co/desh2608/icefall-asr-spgispeech-pruned-transducer-stateless2; cd ..

path_to_pretrained_asr_model="/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pretrained/icefall-asr-spgispeech-pruned-transducer-stateless2"
# ln -s $path_to_pretrained_asr_model/exp/pretrained.pt $path_to_pretrained_asr_model/exp/epoch-1.pt
lang=$path_to_pretrained_asr_model/data/lang_bpe_500/

scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless2/*.* pruned_transducer_stateless2/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless2_context/*.* pruned_transducer_stateless2_context/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/*.* ruizhe_contextual/.

scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/rare_words data/.

#### re-use fbank for un-normalized text
cd /exp/rhuang/icefall_latest/egs/spgispeech/ASR/
mamba activate /exp/rhuang/mambaforge/envs/icefall2
python ruizhe_contextual/get_train_cuts.py  # This seems not working!
python ruizhe_contextual/get_train_cuts2.py

for f in /exp/rhuang/icefall/egs/spgispeech/ASR/data/fbank_no_norm/feats_train_*; do
    ln -s $(realpath $f) data/fbank/.
done

#### prepare data from scratch -- it has to be done on GPU!
# egs/spgispeech/ASR/prepare.sh
python local/compute_fbank_spgispeech.py --train --num-splits 20 --start 2
# --start
# --stop

# egs/spgispeech/ASR/local/compute_fbank_spgispeech.py
# Set: output_dir = Path("data/fbank_temp")

scp /exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/manifests/cuts_{dev,val}.jsonl.gz \
  rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/.

# https://lhotse.readthedocs.io/en/latest/_modules/lhotse/audio.html
# class AudioSource:
# https://stackoverflow.com/questions/49908399/replace-attributes-in-data-class-objects
cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
part="dev"
part="val"
python -c """
from lhotse import CutSet
from dataclasses import replace

file_name = '/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz'
cuts = CutSet.from_file(file_name)
print(f'len(cuts) = {len(cuts)}')
s1 = '/exp/rhuang/icefall_latest/egs/spgispeech/ASR/download/spgispeech/'
s2 = '/export/c01/corpora6/spgispeech/spgispeech_recovered_uncomplete/'
for c in cuts:
    # for r in c.recording.sources:
    #     r.source = r.source.replace(s1, s2)
    assert len(c.recording.sources) == 1
    ss = c.recording.sources[0].source.replace(s1, s2)
    c.recording.sources[0] = replace(c.recording.sources[0], source=ss)

file_name = '/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}_.jsonl.gz'
cuts.to_file(file_name)
print(f'Done: {file_name}')
"""
# It seems the above doesn't work.
# Use the following instead
s1='/exp/rhuang/icefall_latest/egs/spgispeech/ASR/download/spgispeech/'
s2='/export/c01/corpora6/spgispeech/spgispeech_recovered_uncomplete/'
zcat /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz |\
  sed "s%$s1%$s2%g" | gzip \
> /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}_.jsonl.gz
mv /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}_.jsonl.gz \
  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz

# kaldifeat cannot be installed correctly...
if False:
    # https://github.com/search?q=repo%3Acsukuangfj%2Fkaldifeat%20CMAKE_ARGS&type=code
    # https://github.com/csukuangfj/kaldifeat/blob/master/doc/source/installation.rst

    CUDNN_LIBRARY_PATH=/home/smielke/cuda-cudnn/lib64
    CUDNN_INCLUDE_PATH=/home/smielke/cuda-cudnn/include
    CUDA_TOOLKIT_DIR=/usr/local/cuda

    export KALDIFEAT_MAKE_ARGS="-j4"
    export KALDIFEAT_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCUDNN_LIBRARY_PATH=$CUDNN_LIBRARY_PATH/libcudnn.so -DCUDNN_INCLUDE_PATH=$CUDNN_INCLUDE_PATH"
    # pip install --verbose kaldifeat
    cd /export/fs04/a12/rhuang/kaldifeat/
    python setup.py install

    # https://csukuangfj.github.io/kaldifeat/installation.html#install-kaldifeat-from-conda-only-for-linux
    mamba install -c kaldifeat -c pytorch -c conda-forge kaldifeat python=3.8 cudatoolkit=10.2 pytorch=1.12.1
    mamba install -c kaldifeat -c pytorch cpuonly kaldifeat python=3.8 pytorch=1.12.1

    python3 -c "import kaldifeat; print(kaldifeat.__version__)"

    # still have
    conda activate /export/fs04/a12/rhuang/anaconda/anaconda3/envs/espnet_gpu
    mamba activate /home/rhuang/mambaforge/envs/efrat2

    export PYTHONPATH=/export/fs04/a12/rhuang/kaldifeat/build/lib:/export/fs04/a12/rhuang/kaldifeat/kaldifeat/python:$PYTHONPATH

mkdir -p data/fbank
python local/compute_fbank_spgispeech.py --test

# Don't use kaldifeat!!!
# Checkout other recipes of how they compute fbank

python -c """
from pathlib import Path
output_dir = Path("data/fbank")

sampling_rate = 16000
num_mel_bins = 80


"""