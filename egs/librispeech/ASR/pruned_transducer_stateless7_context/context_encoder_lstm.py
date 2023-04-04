import torch
from context_encoder import ContextEncoder

class ContextEncoderLSTM(ContextEncoder):
    def __init__(
        self,
        vocab_size: int = None,
        context_encoder_dim: int = None,
        output_dim: int = None,
        num_layers: int = None,
        num_directions: int = None,
        drop_out: float = None,
    ):
        super(ContextEncoderLSTM, self).__init__()
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.context_encoder_dim = context_encoder_dim

        self.embed = torch.nn.Embedding(
            vocab_size, 
            context_encoder_dim
        )
        self.rnn = torch.nn.LSTM(
            input_size=context_encoder_dim, 
            hidden_size=context_encoder_dim, 
            num_layers=self.num_layers,
            batch_first=True, 
            bidirectional=(self.num_directions == 2), 
            dropout=0.1 if self.num_layers > 1 else 0
        )
        self.linear = torch.nn.Linear(
            context_encoder_dim * self.num_directions, 
            output_dim
        )

        # TODO: Do we need some relu layer?
        # https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        word_list, 
        word_lengths,
    ):
        out = self.embed(word_list)
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        out = torch.nn.utils.rnn.pack_padded_sequence(
            out, 
            batch_first=True, 
            lengths=word_lengths, 
            enforce_sorted=False
        )
        output, (hn, cn) = self.rnn(out)  # use default all zeros (h_0, c_0)

        # # https://discuss.pytorch.org/t/bidirectional-3-layer-lstm-hidden-output/41336/4
        # final_state = hn.view(
        #     self.num_layers, 
        #     self.num_directions,
        #     word_list.shape[0], 
        #     self.encoder_dim,
        # )[-1]  # Only the last layer
        # h_1, h_2 = final_state[0], final_state[1]
        # # X = h_1 + h_2                     # Add both states (needs different input size for first linear layer)
        # final_h = torch.cat((h_1, h_2), dim=1)  # Concatenate both states
        # final_h = self.linear(final_h)

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer.
        # hidden[-2, :, : ] is the last of the forwards RNN 
        # hidden[-1, :, : ] is the last of the backwards RNN
        h_1, h_2 = hn[-2, :, : ] , hn[-1, :, : ]
        final_h = torch.cat((h_1, h_2), dim=1)  # Concatenate both states
        final_h = self.linear(final_h)

        return final_h
