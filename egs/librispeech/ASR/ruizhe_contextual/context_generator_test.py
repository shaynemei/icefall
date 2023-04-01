from egs.librispeech.ASR.pruned_transducer_stateless7_context.context_generator import ContextGenerator

import logging
import argparse
from pathlib import Path
import sentencepiece as spm
from itertools import chain

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

def parse_opts():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--context-dir",
        type=str,
        default="data/fbai-speech/is21_deep_bias/",
        help="",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


def main(params):
    logging.info("About to load context generator")
    params.context_dir = Path(params.context_dir)
    params.lang_dir = Path(params.lang_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(str(params.lang_dir / "bpe.model"))

    context_generator = ContextGenerator(
        path_is21_deep_bias=params.context_dir,
        sp=sp,
        is_predefined=True,
        n_distractors=100,
        keep_ratio=1.0,
        is_full_context=False,
    )
    
    for uid, context_rare_words in chain(
        context_generator.test_clean_biasing_list.items(),
        context_generator.test_other_biasing_list.items(),
    ):
        # import pdb; pdb.set_trace()
        for w in context_rare_words:
            if w in context_generator.common_words:
                logging.warning(f"{uid} {w}")

if __name__ == '__main__':
    opts = parse_opts()

    main(opts)
