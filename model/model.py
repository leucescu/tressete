import torch.nn as nn

class TressetteMLP(nn.Module):
    def __init__(self, input_dim=204, hidden_dim=512, output_dim=256, dropout_rate=0.3):
        super().__init__()
        
        # Input processing with batch normalization
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        
        # Deep residual blocks
        self.block1 = self._make_residual_block(hidden_dim, dropout_rate)
        self.block2 = self._make_residual_block(hidden_dim, dropout_rate)
        self.block3 = self._make_residual_block(hidden_dim, dropout_rate)
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def _make_residual_block(self, dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        # Input processing
        x = self.input_net(x)
        
        # Residual blocks
        residual = x
        x = self.block1(x) + residual
        
        residual = x
        x = self.block2(x) + residual
        
        residual = x
        x = self.block3(x) + residual
        
        # Output processing
        return self.output_net(x)