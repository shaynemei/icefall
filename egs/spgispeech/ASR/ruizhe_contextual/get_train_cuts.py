from lhotse import CutSet
import string
from tqdm import tqdm

# https://github.com/lhotse-speech/lhotse/blob/172e9f7a09e70cee4ee977fa9d288f96616cf7ec/lhotse/recipes/spgispeech.py
def normalize(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text

in_file_name = "/exp/rhuang/icefall/egs/spgispeech/ASR/data/manifests_no_norm/cuts_train_shuf.jsonl.gz"
out_file_name = "/exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/manifests/cuts_train_shuf.jsonl.gz"

print(f"Loading cuts from: {in_file_name}")
cuts = CutSet.from_file(in_file_name)

print(f"len(cuts) = {len(cuts)}")  # len(cuts) = 5886320
for c in tqdm(cuts):
    for s in c.supervisions:
        s.text = normalize(s.text)

print(f"Saving cuts to: {out_file_name}")
# cuts = cuts.trim_to_supervisions(keep_overlapping=False).to_eager()
# cuts = cuts.sort_by_duration()
cuts.to_file(out_file_name)
print("Done.")
