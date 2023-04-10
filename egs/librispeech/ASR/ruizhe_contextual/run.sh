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

# librispeech+gigaspeech
git clone https://huggingface.co/WeijiZhuang/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02
path_to_pretrained_asr_model="/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pretrained/icefall-asr-librispeech-pruned-transducer-stateless8-2022-12-02"
lang=$path_to_pretrained_asr_model/data/lang_bpe_500/

scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/*.* pruned_transducer_stateless7_context/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/*.* ruizhe_contextual/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless2/beam_search.py pruned_transducer_stateless2/.

scp -r pruned_transducer_stateless7_context/exp/exp_libri_full/epoch-{1,2}*.pt rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full/.
scp -r pruned_transducer_stateless7_context/exp/exp_libri_full_c100_continue3/epoch-10.pt rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c100_continue3/.

cd /exp/rhuang/icefall_latest/egs/librispeech/ASR/
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2_10pt/
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_no_stage1
scp -r $exp_dir/epoch-10.pt rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/$exp_dir/.

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




5.54

5.27
5.29
5.39
5.4
5.27
5.42
5.28
5.3
5.32
5.34
5.37
5.51
5.35
5.37
5.29
5.36
5.37
5.37
5.2
5.28
5.44
5.33
5.42
5.36
5.37
5.27
5.32
5.47
5.32
5.36
5.39
5.35
5.56
5.44


cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/

hyp_in=pruned_transducer_stateless7_context/exp/exp_libri_full_c100_continue3/modified_beam_search_LODR/recogs-test-clean-epoch-17-avg-1-modified_beam_search_LODR-beam-size-4-rnnlm-lm-scale-0.4-LODR-2gram-scale--0.16.txt
hyp=pruned_transducer_stateless7_context/exp/exp_libri_full_c100_continue3/modified_beam_search_LODR/recogs-test-clean-epoch-17-avg-1-modified_beam_search_LODR-beam-size-4-rnnlm-lm-scale-0.4-LODR-2gram-scale--0.16.hyp.txt
ref=data/fbai-speech/is21_deep_bias/ref/test-clean.biasing_100.tsv
python ruizhe_contextual/recogs_to_text.py \
  --cuts data/fbank/librispeech_cuts_test-clean.jsonl.gz \
  --input $hyp_in \
  --out $hyp

wc $ref $hyp
python data/fbai-speech/is21_deep_bias/score.py \
  --refs $ref \
  --hyps $hyp


--biased-lm-scale 
8 1.93(1.23/7.59), 4.72(3.26/17.57)
9 1.91(1.24/7.38), 4.67(3.25/17.18)  <--- best
10 1.92(1.26/7.20), 4.68(3.27/17.05)
11 1.97(1.33/7.19), 4.66(3.29/16.69)
12 2.02(1.39/7.12), 4.69(3.35/16.49)

--biased-lm-scale + neural (rd)  (mismatched context collector)
10(pd) 1.83(1.44/4.98), 4.09(3.30/11.07)
11 1.82(1.46/4.76), 4.15(3.37/10.93)
10 1.68(1.28/4.88), 4.10(3.30/11.20)
9 1.55(1.17/4.67), 4.01(3.19/11.23) <--- best
8 1.56(1.15/4.89), 4.04(3.21/11.36)
7 1.64(1.20/5.24), 4.04(3.15/11.83)
6 1.61(1.15/5.33), 4.14(3.20/12.37)
5 1.63(1.16/5.45), 4.13(3.17/12.56)

--biased-lm-scale + neural (rd)  (matched context collector)
15 1.75(1.34/5.03), 3.96(3.25/10.21)
13 1.72(1.31/5.03), 3.93(3.20/10.34)
12 1.66(1.24/5.05), 3.95(3.17/10.77)
11 1.64(1.20/5.22), 3.90(3.12/10.77) <--- best
10 1.64(1.20/5.21), 3.96(3.17/10.82)
9 1.60(1.16/5.17), 3.96(3.16/10.97)
8 1.63(1.17/5.43), 3.95(3.14/11.12)
7 1.64(1.18/5.38), 3.96(3.11/11.42)

all
11 1.49(1.10/4.69), 3.56(2.84/9.91)
