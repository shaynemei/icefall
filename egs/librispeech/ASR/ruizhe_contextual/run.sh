# extract a subset of libri100
cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
ipython

import lhotse
from lhotse import load_manifest, CutSet
cuts = load_manifest("data/fbank/librispeech_cuts_train-clean-100.jsonl.gz")
random_sample = cuts.sample(n_cuts=int(len(cuts) * 0.4))
cuts_ = [c for c in cuts if c in random_sample]
print(len(cuts_))
cuts_ = CutSet.from_items(cuts_)
cuts_.to_file("data/fbank/librispeech_cuts_train-clean-100-0.4.jsonl.gz")

# In [6]: random_sample.describe()
# Cut statistics:
# ╒═══════════════════════════╤═══════════╕
# │ Cuts count:               │ 34246     │
# ├───────────────────────────┼───────────┤
# │ Total duration (hh:mm:ss) │ 121:35:57 │
# ├───────────────────────────┼───────────┤
# │ mean                      │ 12.8      │
# ├───────────────────────────┼───────────┤
# │ std                       │ 3.8       │
# ├───────────────────────────┼───────────┤
# │ min                       │ 1.6       │
# ├───────────────────────────┼───────────┤
# │ 25%                       │ 11.4      │
# ├───────────────────────────┼───────────┤
# │ 50%                       │ 13.8      │
# ├───────────────────────────┼───────────┤
# │ 75%                       │ 15.3      │
# ├───────────────────────────┼───────────┤
# │ 99%                       │ 18.1      │
# ├───────────────────────────┼───────────┤
# │ 99.5%                     │ 18.4      │
# ├───────────────────────────┼───────────┤
# │ 99.9%                     │ 18.9      │
# ├───────────────────────────┼───────────┤
# │ max                       │ 20.3      │
# ├───────────────────────────┼───────────┤
# │ Recordings available:     │ 34246     │
# ├───────────────────────────┼───────────┤
# │ Features available:       │ 34246     │
# ├───────────────────────────┼───────────┤
# │ Supervisions available:   │ 34246     │
# ╘═══════════════════════════╧═══════════╛
# Speech duration statistics:
# ╒══════════════════════════════╤═══════════╤══════════════════════╕
# │ Total speech duration        │ 121:35:57 │ 100.00% of recording │
# ├──────────────────────────────┼───────────┼──────────────────────┤
# │ Total speaking time duration │ 121:35:57 │ 100.00% of recording │
# ├──────────────────────────────┼───────────┼──────────────────────┤
# │ Total silence duration       │ 00:00:00  │ 0.00% of recording   │
# ╘══════════════════════════════╧═══════════╧══════════════════════╛

cd /exp/rhuang/icefall_latest/egs/librispeech/ASR
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7/*.* pruned_transducer_stateless7/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/tdnn_lstm_ctc/asr_datamodule.py tdnn_lstm_ctc/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/*.* ruizhe_contextual/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless2/beam_search.py pruned_transducer_stateless2/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-clean-100-0.4.jsonl.gz data/fbank/.


cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/data
git clone git@github.com:facebookresearch/fbai-speech.git
cd -

cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/
export PATH=/export/fs04/a12/rhuang/git-lfs-3.2.0/:$PATH
git lfs install
git lfs version
mkdir -p pretrained
cd pretrained; git clone https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11; cd ..
path_to_pretrained_asr_model="/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pretrained/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/"
# ln -s $path_to_pretrained_asr_model/exp/pretrained.pt $path_to_pretrained_asr_model/exp/epoch-1.pt
lang=$path_to_pretrained_asr_model/data/lang_bpe_500/

ln -sf $path_to_pretrained_asr_model/exp/epoch-30.pt pruned_transducer_stateless7_context/exp/exp_libri_full/epoch-1.pt

scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/*.* pruned_transducer_stateless7_context/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/*.* ruizhe_contextual/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless2/beam_search.py pruned_transducer_stateless2/.

scp -r pruned_transducer_stateless7_context/exp/exp_libri_full/epoch-{1,2}*.pt rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full/.

# clsp grid initialize
# ql
cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
mamba activate /home/rhuang/mambaforge/envs/efrat2
export PYTHONPATH=/export/fs04/a12/rhuang/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/export/fs04/a12/rhuang/k2/build/temp.linux-x86_64-cpython-38/lib/:$PYTHONPATH # for `import _k2`
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH
source /home/gqin2/scripts/acquire-gpu 1


mamba install transformers

# get ratio of rare words
python -c '''
from lhotse import SupervisionSet
from itertools import chain
with open("data/fbai-speech/is21_deep_bias/words/common_words_5k.txt", "r") as fin:
  common_words = set([l.strip().upper() for l in fin if len(l) > 0])  # a list of strings
print(f"Number of common words: {len(common_words)}")
sups1 = SupervisionSet.from_file("data/manifests/librispeech_supervisions_train-clean-100.jsonl.gz")
sups2 = SupervisionSet.from_file("data/manifests/librispeech_supervisions_train-clean-360.jsonl.gz")
sups3 = SupervisionSet.from_file("data/manifests/librispeech_supervisions_train-other-500.jsonl.gz")
all_cnt = 0
rare_cnt = 0 
for sup in chain(sups1, sups2, sups3):
  ws = sup.text.split()
  rs = [w for w in ws if w not in common_words]
  all_cnt += len(ws)
  rare_cnt += len(rs)
print(f"all_cnt={all_cnt}")
print(f"rare_cnt={rare_cnt}")
print(f"ratio={rare_cnt/all_cnt:.2f}")
'''
