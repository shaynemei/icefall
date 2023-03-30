import torch

class ContextEncoderLSTM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int,
        output_dim: int,
        num_layers: int,
        num_directions: int,
    ):
        super(ContextEncoderLSTM, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.encoder_dim = encoder_dim

        self.embed = torch.nn.Embedding(
            vocab_size, 
            encoder_dim
        )
        self.rnn = torch.nn.LSTM(
            input_size=encoder_dim, 
            hidden_size=encoder_dim, 
            num_layers=self.num_layers,
            batch_first=True, 
            bidirectional=(self.num_directions == 2), 
            dropout=0.1 if self.num_layers > 1 else 0
        )
        self.linear = torch.nn.Linear(
            encoder_dim * self.num_directions, 
            output_dim
        )

        # TODO: Do we need some relu layer?
        # https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x, 
        lengths
    ):
        out = self.embed(x)
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, 
            batch_first=True, 
            lengths=lengths, 
            enforce_sorted=False
        )
        out = self.rnn(out)  # use default all zeros (h_0, c_0)
        return out

    def embed_contexts(
        self, 
        word_list,
        word_lengths,
        num_words_per_utt,
    ):
        """
        Args:
            word_list: 
                A list of words, where each word is a list of token ids.
                The list of tokens for each word has been padded.
            word_lengths:
                The number of tokens per word
            num_words_per_utt:
                The number of words in the context for each utterance
        Returns:
            final_h:
                A tensor of shape (batch_size, max(num_words_per_utt) + 1, joiner_dim),
                which is the embedding for each context word.
            mask_h:
                A tensor of shape (batch_size, max(num_words_per_utt) + 1),
                which contains a True/False mask for final_h
        """

        # print(f"word_list.shape={word_list.shape}")
        output, (hn, cn) = self.forward(word_list, word_lengths)

        # https://discuss.pytorch.org/t/bidirectional-3-layer-lstm-hidden-output/41336/4
        final_state = hn.view(
            self.num_layers, 
            self.num_directions,
            word_list.shape[0], 
            self.encoder_dim,
        )[-1]  # Only the last layer
        h_1, h_2 = final_state[0], final_state[1]
        # X = h_1 + h_2                     # Add both states (needs different input size for first linear layer)
        final_h = torch.cat((h_1, h_2), dim=1)  # Concatenate both states
        final_h = self.linear(final_h)

        # # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # # outputs are always from the last layer.
        # # hidden[-2, :, : ] is the last of the forwards RNN 
        # # hidden[-1, :, : ] is the last of the backwards RNN
        # h_1, h_2 = hn[-2, :, : ] , hn[-1, :, : ]
        # final_h = torch.cat((h_1, h_2), dim=1)  # Concatenate both states
        # final_h = self.linear(final_h)

        final_h = torch.split(final_h, num_words_per_utt)
        final_h = torch.nn.utils.rnn.pad_sequence(
            final_h, 
            batch_first=True, 
            padding_value=0.0
        )
        # print(f"final_h.shape={final_h.shape}")

        # add one no-bias token
        no_bias_h = torch.zeros(final_h.shape[0], 1, final_h.shape[-1])
        no_bias_h = no_bias_h.to(final_h.device)
        final_h = torch.cat((no_bias_h, final_h), dim=1)
        # print(final_h)

        # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
        mask_h = torch.arange(max(num_words_per_utt) + 1)
        mask_h = mask_h.expand(len(num_words_per_utt), max(num_words_per_utt) + 1) > torch.Tensor(num_words_per_utt).unsqueeze(1)
        mask_h = mask_h.to(final_h.device)

        return final_h, mask_h

    def clustering(self):
        pass

    def cache(self):
        pass
