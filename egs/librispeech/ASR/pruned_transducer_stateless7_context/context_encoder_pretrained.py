import torch
from context_encoder import ContextEncoder
import torch.nn.functional as F


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
        self.linear1 = torch.nn.Linear(
            context_encoder_dim,
            context_encoder_dim,
        )
        self.linear2 = torch.nn.Linear(
            context_encoder_dim,
            output_dim
        )

    def forward(
        self, 
        word_list, 
        word_lengths,
    ):
        out = word_list  # Shape: N*L*D
        out = self.drop_out(out)
        out = F.relu(self.linear1(out))
        out = self.drop_out(out)
        out = F.relu(self.linear2(out))
        return out
