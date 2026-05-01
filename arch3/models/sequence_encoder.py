import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MAX_SEQ_LEN = 512
EVENT_DIM = 3
HIDDEN_DIM = 64
N_LAYERS = 2


def _make_autoencoder():
    if not TORCH_AVAILABLE:
        return None

    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim=EVENT_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS):
            super().__init__()
            drop = 0.2 if n_layers > 1 else 0.0
            self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers,
                                   batch_first=True, dropout=drop)
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                                   batch_first=True, dropout=drop)
            self.output_proj = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            seq_len = x.size(1)
            _, (h, c) = self.encoder(x)
            latent = h[-1]
            dec_in = latent.unsqueeze(1).repeat(1, seq_len, 1)
            dec_out, _ = self.decoder(dec_in, (h, c))
            return self.output_proj(dec_out), latent

    return LSTMAutoencoder


class SequenceEncoder:
    def __init__(self, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                 epochs=30, batch_size=32, lr=1e-3, device=None):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.activity_vocab = {}
        self.is_fitted = False

    def _build_vocab(self, sequences):
        activities = set()
        for seq in sequences:
            for event in seq:
                activities.add(event[0])
        self.activity_vocab = {a: i + 1 for i, a in enumerate(sorted(activities))}
        self.activity_vocab['<PAD>'] = 0

    def _encode_sequence(self, sequence):
        encoded = []
        for activity, hour, weekday in sequence:
            act_id = self.activity_vocab.get(activity, 0)
            encoded.append([
                act_id / max(len(self.activity_vocab), 1),
                hour / 23.0,
                weekday / 6.0,
            ])
        return encoded

    def _sequences_to_tensor(self, sequences):
        if not TORCH_AVAILABLE:
            return None
        tensors = []
        for seq in sequences:
            encoded = self._encode_sequence(seq)
            if len(encoded) > MAX_SEQ_LEN:
                encoded = encoded[-MAX_SEQ_LEN:]
            pad_len = MAX_SEQ_LEN - len(encoded)
            if pad_len > 0:
                encoded = [[0.0, 0.0, 0.0]] * pad_len + encoded
            tensors.append(encoded)
        return torch.tensor(tensors, dtype=torch.float32)

    def fit(self, sequences):
        if not TORCH_AVAILABLE:
            print("  torch не установлен — LSTM пропущен")
            self.is_fitted = True
            return self

        self._build_vocab(sequences)
        X = self._sequences_to_tensor(sequences).to(self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        AutoencoderClass = _make_autoencoder()
        self.model = AutoencoderClass(
            input_dim=EVENT_DIM, hidden_dim=self.hidden_dim, n_layers=self.n_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  LSTM epoch {epoch+1}/{self.epochs}, loss={total_loss/len(loader):.4f}")

        self.is_fitted = True
        return self

    def extract_features(self, sequences):
        n = len(sequences)
        if not TORCH_AVAILABLE or not self.is_fitted or self.model is None:
            return np.zeros((n, 4), dtype=np.float32)

        self.model.eval()
        X = self._sequences_to_tensor(sequences).to(self.device)
        features = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size]
                recon, latent = self.model(batch)
                recon_error = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
                lat = latent.cpu().numpy()
                batch_feats = np.column_stack([
                    recon_error,
                    lat[:, 0],
                    lat[:, 1] if lat.shape[1] > 1 else np.zeros(len(lat)),
                    lat.std(axis=1),
                ])
                features.append(batch_feats)
        return np.vstack(features).astype(np.float32)

    def get_feature_names(self):
        return ['lstm_recon_error', 'lstm_latent_0', 'lstm_latent_1', 'lstm_latent_std']
