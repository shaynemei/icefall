cd /exp/rhuang/icefall_latest/egs/spgispeech/ASR/
# export PATH=/export/fs04/a12/rhuang/git-lfs-3.2.0/:$PATH
git lfs install
git lfs version
mkdir -p pretrained
cd pretrained; git clone https://huggingface.co/desh2608/icefall-asr-spgispeech-pruned-transducer-stateless2; cd ..

path_to_pretrained_asr_model="/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pretrained/icefall-asr-spgispeech-pruned-transducer-stateless2"
# ln -s $path_to_pretrained_asr_model/exp/pretrained.pt $path_to_pretrained_asr_model/exp/epoch-1.pt
lang=$path_to_pretrained_asr_model/data/lang_bpe_500/

scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless2_context/*.* pruned_transducer_stateless2_context/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/*.* ruizhe_contextual/.

#### re-use fbank for un-normalized text
cd /exp/rhuang/icefall_latest/egs/spgispeech/ASR/
python ruizhe_contextual/get_train_cuts.py

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