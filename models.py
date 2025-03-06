import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ----------------------------
# Helper Modules
# ----------------------------
class GLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.a = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.b = nn.Linear(input_size, input_size)
    
    def forward(self, x):
        gate = self.sigmoid(self.b(x))
        x = self.a(x)
        return gate * x

class TemporalLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        t, n, h = x.size()
        x = x.view(t * n, h)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

class TemporalConvLayer(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
    
    def forward(self, x):
        T, N, H = x.size()
        x = x.permute(1, 2, 0)  # (N, H, T)
        x = self.conv(x)        # (N, output_size, T)
        x = self.dropout(x)
        x = x.permute(2, 0, 1)  # (T, N, output_size)
        x = self.layer_norm(x)
        return x

class GateResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, context_size=None, is_temporal=True, use_conv=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_temporal = is_temporal
        self.use_conv = use_conv
        
        if self.is_temporal:
            if self.input_size != self.output_size:
                self.skip_layer = TemporalLayer(nn.Linear(self.input_size, self.output_size))
            if self.context_size is not None:
                self.c = TemporalLayer(nn.Linear(self.context_size, self.hidden_size, bias=False))
            self.dense1 = TemporalLayer(nn.Linear(self.input_size, self.hidden_size))
            self.elu = nn.ELU()
            self.dense2 = TemporalLayer(nn.Linear(self.hidden_size, self.output_size))
            if self.use_conv:
                self.conv = TemporalConvLayer(self.output_size, self.output_size, kernel_size=3, dropout=dropout)
            self.dropout_layer = nn.Dropout(self.dropout)
            self.gate = TemporalLayer(GLU(self.output_size))
            self.layer_norm = nn.LayerNorm(self.output_size)
        else:
            if self.input_size != self.output_size:
                self.skip_layer = nn.Linear(self.input_size, self.output_size)
            if self.context_size is not None:
                self.c = nn.Linear(self.context_size, self.hidden_size, bias=False)
            self.dense1 = nn.Linear(self.input_size, self.hidden_size)
            self.elu = nn.ELU()
            self.dense2 = nn.Linear(self.hidden_size, self.output_size)
            self.dropout_layer = nn.Dropout(self.dropout)
            self.gate = GLU(self.output_size)
            self.layer_norm = nn.LayerNorm(self.output_size)
    
    def forward(self, x, c=None):
        if self.input_size != self.output_size:
            a = self.skip_layer(x)
        else:
            a = x
        x = self.dense1(x)
        if c is not None and self.context_size is not None:
            c = self.c(c.unsqueeze(1))
            x += c
        eta_2 = self.elu(x)
        eta_1 = self.dense2(eta_2)
        if self.is_temporal and self.use_conv:
            conv_out = self.conv(eta_1)
            eta_1 = eta_1 + conv_out
        eta_1 = self.dropout_layer(eta_1)
        gate = self.gate(eta_1)
        gate += a
        x = self.layer_norm(gate)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + query)
        return attn_output, attn_weights

# ----------------------------
# TFTPredictor Model
# ----------------------------
class TFTPredictor(nn.Module):
    def __init__(self, input_dim, static_dim, hidden_dim, output_size, dropout=0.1, learning_rate=1e-3, num_heads=4):
        super().__init__()
        self.static_grn = GateResidualNetwork(static_dim, hidden_dim, hidden_dim, dropout, is_temporal=False)
        self.variable_selector = GateResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, is_temporal=True, use_conv=True)
        self.seq_transform = GateResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, is_temporal=True, use_conv=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.multihead_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.grn1 = GateResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout, is_temporal=True, use_conv=True)
        self.grn2 = GateResidualNetwork(hidden_dim, hidden_dim, output_size, dropout, is_temporal=False)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, static_x, seq_x, return_attn=False):
        static_features = self.static_grn(static_x)
        variable_scores = self.variable_selector(seq_x)
        seq_features = self.seq_transform(seq_x) * variable_scores
        lstm_out, _ = self.lstm(seq_features)
        attention_output, attn_weights = self.multihead_attention(lstm_out, lstm_out, lstm_out)
        combined = attention_output[:, -1, :] + static_features
        x = self.grn1(combined.unsqueeze(1))
        x = self.dropout_layer(x)
        x = self.grn2(x).squeeze(1)
        if return_attn:
            return x, attn_weights
        return x

# ----------------------------
# Training and Testing Helpers
# ----------------------------
def train_torch():
    def custom_train_torch(model, train_loader, epochs, cfg):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (static_x, seq_x, y) in enumerate(train_loader):
                static_x, seq_x, y = static_x.to(device), seq_x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(static_x, seq_x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
            print(f"Epoch {epoch+1} complete. Avg Loss: {epoch_loss/len(train_loader):.4f}")
        model.to("cpu")
        return model
    return custom_train_torch

def test_torch():
    def custom_test_torch(model, test_loader, cfg):
        criterion = nn.BCEWithLogitsLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_loss = 0.0
        correct = 0
        total = 0
        model.to(device)
        model.eval()
        with torch.no_grad():
            for static_x, seq_x, y in test_loader:
                static_x, seq_x, y = static_x.to(device), seq_x.to(device), y.to(device)
                outputs = model(static_x, seq_x)
                loss = criterion(outputs, y)
                total_loss += loss.item()
                preds = (outputs > 0).float()
                correct += (preds == y).sum().item()
                total += y.numel()
        average_loss = total_loss / len(test_loader)
        accuracy = correct / total
        metrics = {"accuracy": accuracy}
        model.to("cpu")
        return average_loss, accuracy, metrics
    return custom_test_torch

# ----------------------------
# Custom Dataset
# ----------------------------
class TFTDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        static_x, seq_x, y = self.sequences[idx]
        static_x = torch.tensor(static_x, dtype=torch.float32)
        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return static_x, seq_x, y
