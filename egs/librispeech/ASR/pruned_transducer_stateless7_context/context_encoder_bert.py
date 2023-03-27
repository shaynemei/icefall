import torch

class ContextEncoder(torch.nn.Module):
  def __init__(self, num_inputs):
    super(ContextEncoder, self).__init__()
    self.num_layers = 1
    self.directions = 2
    self.embed = torch.nn.Embedding(num_inputs, encoder_dim)
    self.rnn = torch.nn.LSTM(input_size=encoder_dim, hidden_size=encoder_dim, num_layers=self.num_layers, batch_first=True, bidirectional=(self.directions == 2), dropout=0.1 if self.num_layers > 1 else 0)
    self.linear = torch.nn.Linear(encoder_dim*self.directions, joiner_dim)

  def forward(self, x, lengths):
    out = x
    out = self.embed(out)    
    # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
    out = torch.nn.utils.rnn.pack_padded_sequence(out, batch_first=True, lengths=lengths, enforce_sorted=False)    
    out = self.rnn(out)
    return out
  
  def embed_contexts(self, c, C_L, STR_L):
    # print(f"c.shape={c.shape}")    
    output, (hn, cn) = self.forward(c, STR_L)

    # https://discuss.pytorch.org/t/bidirectional-3-layer-lstm-hidden-output/41336/4
    batch_size = c.shape[0]
    final_state = hn.view(self.num_layers, self.directions, batch_size, encoder_dim)[-1]  # Only the last layer
    h_1, h_2 = final_state[0], final_state[1]
    # X = h_1 + h_2                     # Add both states (needs different input size for first linear layer)
    final_h = torch.cat((h_1, h_2), 1)  # Concatenate both states
    final_h = self.linear(final_h)

    final_h = torch.split(final_h, C_L)
    final_h = torch.nn.utils.rnn.pad_sequence(final_h, batch_first=True, padding_value=0.0)
    # print(f"final_h.shape={final_h.shape}")

    # add one no-bias token
    no_bias_h = torch.zeros(final_h.shape[0], 1, final_h.shape[-1])
    no_bias_h = no_bias_h.to(final_h.device)
    final_h = torch.cat((no_bias_h, final_h), 1)
    # print(final_h)

    # https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    mask_h = torch.arange(max(C_L) + 1).expand(len(C_L), max(C_L) + 1) > torch.Tensor(C_L).unsqueeze(1)

    return final_h, mask_h

  def clustering(self):
    pass

  def cache(self):
    pass