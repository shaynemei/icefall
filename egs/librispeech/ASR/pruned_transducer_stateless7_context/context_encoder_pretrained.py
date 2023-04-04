import torch
from context_encoder import ContextEncoder


class ContextEncoderPretrained(ContextEncoder):
    def __init__(
        self,
        vocab_size: int = None,
        context_encoder_dim: int = None,
        output_dim: int = None,
        num_layers: int = None,
        num_directions: int = None,
        drop_out: float = 0.3,
    ):
        super(ContextEncoderPretrained, self).__init__()

        self.drop_out = torch.nn.Dropout(drop_out)
        self.linear = torch.nn.Linear(
            context_encoder_dim,
            output_dim
        )
        self.relu = torch.nn.ReLU()

    def forward(
        self, 
        word_list, 
        word_lengths,
    ):
        out = word_list
        out = self.drop_out(out)
        out = self.linear(out)
        out = self.relu(out)
        return out
