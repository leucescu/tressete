import torch.nn as nn

class AttentionMLP(nn.Module):
    def __init__(self, input_dim=162, hidden_dim=256, output_dim=256):  # output_dim = feature dim
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Added extra hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output feature vector
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # (batch, seq_len=1, hidden)
        attn_output, _ = self.attn(x, x, x)
        x = attn_output.squeeze(1)
        return self.fc(x)
