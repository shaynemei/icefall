cd /exp/rhuang/icefall_latest/egs/librispeech/ASR
mamba activate /exp/rhuang/mambaforge/envs/icefall2

python -c '''
from lhotse import load_manifest, CutSet
from itertools import chain
import logging
logging.basicConfig(level=logging.DEBUG)
with open("/exp/rhuang/librispeech/data2/fbai-speech/is21_deep_bias/words/all_words.count.txt", "r") as fin:
  all_words = set([l.strip().upper().split()[0] for l in fin if len(l) > 0])  # a list of strings
logging.info(f"Number of all_words: {len(all_words)}")
basepath = "/exp/rhuang/librispeech/data2/fbank/"
cuts1 = load_manifest(f"{basepath}/librispeech_cuts_train-all-shuf.jsonl.gz")
cuts2 = load_manifest(f"{basepath}/librispeech_cuts_dev-clean.jsonl.gz")
cuts3 = load_manifest(f"{basepath}/librispeech_cuts_dev-other.jsonl.gz")
cuts4 = load_manifest(f"{basepath}/librispeech_cuts_test-clean.jsonl.gz")
cuts5 = load_manifest(f"{basepath}/librispeech_cuts_test-other.jsonl.gz")
for cut in chain(cuts1, cuts2, cuts3, cuts4, cuts5):
  ws = cut.supervisions[0].text.split()
  all_words.update(ws)
logging.info(f"Number of all_words (after): {len(all_words)}")
for w in all_words:
  print(w.lower())
''' > pruned_transducer_stateless7_context/exp/exp_fasttext/fasttext_all_words.txt

time cat pruned_transducer_stateless7_context/exp/exp_fasttext/fasttext_all_words.txt | \
  /exp/rhuang/fastText/fasttext print-word-vectors /exp/rhuang/fastText/cc.en.300.bin \
> pruned_transducer_stateless7_context/exp/exp_fasttext/fasttext_all_words.embeddings.txt

ln -s /exp/rhuang/fastText/cc.en.300.bin pruned_transducer_stateless7_context/exp/exp_fasttext/.