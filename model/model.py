import torch.nn as nn

class TressetteMLP(nn.Module):
    def __init__(self, input_dim=214, hidden_dim=512, output_dim=256, dropout_rate=0.0):  # set dropout=0.0 to test
        super().__init__()

        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed to LayerNorm
            nn.LeakyReLU(0.01)
        )

        self.block1 = self._make_residual_block(hidden_dim)
        self.block2 = self._make_residual_block(hidden_dim)

        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def _make_residual_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.01),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        x = self.input_net(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        return self.output_net(x)
