import torch
import random
from pathlib import Path
import sentencepiece as spm
from typing import List
import logging
import ast

class ContextGenerator(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_is21_deep_bias: Path,
        sp: spm.SentencePieceProcessor,
        context_size: int = 100,
        is_predefined: bool = False,
        keep_ratio: float = 1.0,
    ):
        self.sp = sp
        self.path_is21_deep_bias = path_is21_deep_bias
        self.context_size = context_size
        self.is_predefined = is_predefined
        self.keep_ratio = keep_ratio

        self.all_rare_words2pieces = None
        self.common_words = None
        if not is_predefined:
            with open(path_is21_deep_bias / "words/all_rare_words.txt", "r") as fin:
                all_rare_words = [l.strip().upper() for l in fin if len(l) > 0]  # a list of strings
                all_rare_words_pieces = sp.encode(all_rare_words, out_type=int)  # a list of list of int
                self.all_rare_words2pieces = {w: pieces for w, pieces in zip(all_rare_words, all_rare_words_pieces)}
            
            with open(path_is21_deep_bias / "words/common_words_5k.txt", "r") as fin:
                self.common_words = set([l.strip().upper() for l in fin if len(l) > 0])  # a list of strings

            logging.info(f"Number of common words: {len(self.common_words)}")
            logging.info(f"Number of rare words: {len(self.all_rare_words2pieces)}")
        
        self.test_clean_biasing_list = None
        self.test_other_biasing_list = None
        if is_predefined:
            def read_ref_biasing_list(filename):
                biasing_list = dict()
                with open(filename, "r") as fin:
                    for line in fin:
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        line = line.split("\t")
                        uid, ref_text, ref_rare_words, context_rare_words = line
                        context_rare_words = ast.literal_eval(context_rare_words)
                        biasing_list[uid] = [w.upper() for w in context_rare_words]
                return biasing_list
                    
            self.test_clean_biasing_list = \
                read_ref_biasing_list(self.path_is21_deep_bias / f"ref/test-clean.biasing_{context_size}.tsv")
            self.test_other_biasing_list = \
                read_ref_biasing_list(self.path_is21_deep_bias / f"ref/test-other.biasing_{context_size}.tsv")

            logging.info(f"Number of utterances in test_clean_biasing_list: {len(self.test_clean_biasing_list)}")
            logging.info(f"Number of utterances in test_other_biasing_list: {len(self.test_other_biasing_list)}")

    def get_context_word_list(
        self,
        batch: dict,
    ):
        if self.is_predefined:
            return self.get_context_word_list_predefined(batch=batch)
        else:
            return self.get_context_word_list_random(batch=batch)

    def get_context_word_list_random(
        self,
        batch: dict,
    ):
        """
        Generate context as a list of words for each utterance, given context_size.
        Use keep_ratio to simulate the "imperfect" context which may not have 100% coverage of the ground truth words.
        """
        texts = batch["supervisions"]["text"]

        rare_words_list = []
        distractors_cnt = 0
        for text in texts:
            rare_words = []
            for word in text.split():
                if word not in self.common_words:
                    rare_words.append(word)
                    if word not in self.all_rare_words2pieces:
                        self.all_rare_words2pieces[word] = self.sp.encode(word, out_type=int)
            
            if self.keep_ratio < 1.0 and len(rare_words) > 0:
                rare_words = random.sample(rare_words, int(len(rare_words) * self.keep_ratio))

            distractors_cnt += max(self.context_size - len(rare_words), 0)
            rare_words_list.append(rare_words)
        
        distractors = random.sample(self.all_rare_words2pieces.keys(), distractors_cnt)  # TODO: actually the context should contain both rare and common words
        distractors_pos = 0
        rare_words_pieces_list = []
        max_pieces_len = 0
        for rare_words in rare_words_list:
            n_distractors = max(self.context_size - len(rare_words), 0)
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
        # word_lengths = torch.tensor(word_lengths, dtype=torch.int32)
        # num_words_per_utt = torch.tensor(num_words_per_utt, dtype=torch.int32)

        return word_list, word_lengths, num_words_per_utt

    def get_context_word_list_predefined(
        self,
        batch: dict,
    ):        
        rare_words_list = []
        for cut in batch['supervisions']['cut']:
            uid = cut.supervisions[0].id
            if uid in self.test_clean_biasing_list:
                rare_words_list.append(self.test_clean_biasing_list[uid])
            elif uid in self.test_other_biasing_list:
                rare_words_list.append(self.test_other_biasing_list[uid])
            else:
                logging.error(f"uid={uid} cannot find the predefined biasing list of size {self.context_size}")
        
        rare_words_pieces_list = []
        max_pieces_len = 0
        for rare_words in rare_words_list:
            rare_words_pieces = self.sp.encode(rare_words, out_type=int)
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
        # word_lengths = torch.tensor(word_lengths, dtype=torch.int32)
        # num_words_per_utt = torch.tensor(num_words_per_utt, dtype=torch.int32)

        return word_list, word_lengths, num_words_per_utt
