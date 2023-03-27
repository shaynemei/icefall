import torch
import random
from pathlib import Path
import sentencepiece as spm
from typing import List
import logging

class ContextGenerator(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_is21_deep_bias: Path,
        sp: spm.SentencePieceProcessor,
    ):
        self.sp = sp

        with open(path_is21_deep_bias / "words/all_rare_words.txt") as fin:
            all_rare_words = [l for l in fin if len(l) > 0]  # a list of strings
            all_rare_words_pieces = sp.encode(all_rare_words, out_type=int)  # a list of list of int
            self.all_rare_words2pieces = {w: pieces for w, pieces in zip(all_rare_words, all_rare_words_pieces)}
        
        with open(path_is21_deep_bias / "words/common_words_5k.txt") as fin:
            self.common_words = set([l for l in fin if len(l) > 0])  # a list of strings
        
        logging.info(f"Number of common words: {len(self.common_words)}")
        logging.info(f"Number of rare words: {len(self.all_rare_words2pieces)}")

    def get_context_word_list_random(
        self,
        texts: List[List[str]],
        context_size: int = 100,
        keep_ratio: float = 1.0,
    ):
        """
        Generate context as a list of words for each utterance, given context_size.
        Use keep_ratio to simulate the "imperfect" context which may not have 100% coverage of the ground truth words.
        """
        rare_words_list = []
        distractors_cnt = 0
        for text in texts:
            rare_words = []
            for word in text.split():
                if word not in self.common_words:
                    rare_words.append(word)
                    if word not in self.all_rare_words2pieces:
                        self.all_rare_words2pieces[word] = self.sp.encode(word)
            
            if keep_ratio < 1.0 and len(rare_words) > 0:
                rare_words = random.sample(rare_words, int(len(rare_words) * keep_ratio))

            distractors_cnt += max(context_size - len(rare_words), 0)
            rare_words_list.append(rare_words)
        
        distractors = random.sample(self.all_rare_words2pieces.keys(), distractors_cnt)  # TODO: actually the context should contain both rare and common words
        distractors_pos = 0
        rare_words_pieces_list = []
        max_pieces_len = 0
        for rare_words in rare_words_list:
            n_distractors = max(context_size - len(rare_words), 0)
            if n_distractors > 0:
                rare_words.extend(distractors[distractors_pos: distractors_pos + n_distractors])
                distractors_pos += n_distractors

            rare_words_pieces = [self.all_rare_words2pieces[w] for w in rare_words]
            max_pieces_len = max(max_pieces_len, max(len(pieces) for pieces in rare_words_pieces))
            rare_words_pieces_list.append(rare_words_pieces)

        word_list = []
        word_lengths = []
        num_words_per_utt = []
        pad_token = 0
        for rare_words_pieces in rare_words_pieces_list:
            num_words_per_utt.append(len(rare_words_pieces))
            word_lengths.extend([len(pieces) for pieces in rare_words_pieces])

            for pieces in rare_words_pieces:
                pieces += [pad_token] * (max_pieces_len - len(pieces))
            word_list.extend(rare_words_pieces)

        word_list = torch.tensor(word_list, dtype=torch.int32)
        word_lengths = torch.tensor(word_lengths, dtype=torch.int32)
        # num_words_per_utt = torch.tensor(num_words_per_utt, dtype=torch.int32)

        return word_list, word_lengths, num_words_per_utt
