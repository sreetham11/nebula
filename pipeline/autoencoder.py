"""
Core anomaly model: a GRU/LSTM sequence autoencoder trained ONLY on time
windows from units that never appear in fault_log ("healthy-only"
training). At inference/scoring time it is run over every unit (healthy and
faulty); reconstruction error on faulty units should rise as their signals
drift away from the healthy manifold the model learned.

Architecture: encoder GRU -> latent vector (from final hidden state) ->
decoder GRU (fed the repeated latent vector at every step) -> linear output
projecting back to the original signal dimensions. This "repeated latent as
decoder input" pattern avoids needing teacher forcing at inference and is a
standard simple choice for sequence autoencoders.

Multivariate input per subsystem (see schema_config.SUBSYSTEMS):
  door:  [cycle_time_sec, motor_current_amps]
  bogie: [temperature_c, vibration_rms]
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from schema_config import SUBSYSTEMS
from model_config import (
    AE_ARCHITECTURE, AE_WINDOW_LENGTH, AE_WINDOW_STRIDE, AE_HIDDEN_SIZE,
    AE_LATENT_SIZE, AE_NUM_LAYERS, AE_DROPOUT, AE_BATCH_SIZE, AE_EPOCHS,
    AE_LEARNING_RATE, AE_EARLY_STOP_PATIENCE, AE_SCORING_STRIDE,
    RANDOM_SEED,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SequenceAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=AE_HIDDEN_SIZE, latent_size=AE_LATENT_SIZE,
                 num_layers=AE_NUM_LAYERS, dropout=AE_DROPOUT, architecture=AE_ARCHITECTURE):
        super().__init__()
        rnn_cls = nn.LSTM if architecture == "lstm" else nn.GRU
        self.architecture = architecture
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.encoder_rnn = rnn_cls(
            n_features, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.enc_to_latent = nn.Linear(hidden_size, latent_size)
        self.latent_to_dec_h = nn.Linear(latent_size, hidden_size)
        self.decoder_rnn = rnn_cls(
            latent_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        seq_len = x.size(1)
        _, h_enc = self._run_encoder(x)
        h_last = h_enc[0][-1] if self.architecture == "lstm" else h_enc[-1]
        latent = self.enc_to_latent(h_last)

        dec_h0 = self.latent_to_dec_h(latent).unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)

        if self.architecture == "lstm":
            dec_c0 = torch.zeros_like(dec_h0)
            dec_out, _ = self.decoder_rnn(decoder_input, (dec_h0, dec_c0))
        else:
            dec_out, _ = self.decoder_rnn(decoder_input, dec_h0)

        recon = self.output_layer(dec_out)
        return recon

    def _run_encoder(self, x):
        return self.encoder_rnn(x)


def fit_scaler(df, subsystem_type, healthy_ids):
    cfg = SUBSYSTEMS[subsystem_type]
    healthy_df = df[df[cfg["id_col"]].isin(healthy_ids)]
    scaler = StandardScaler()
    scaler.fit(healthy_df[cfg["signal_cols"]].values)
    return scaler


def build_windows(df, subsystem_type, unit_ids, scaler, window_length, stride):
    """
    Slices each unit's own (already time-sorted) sequence into windows.
    Returns:
      X            : (n_windows, window_length, n_features) scaled array
      end_index    : original dataframe index of each window's last reading
                      (used to attribute a score back to a specific row)
      unit_of_win  : unit id for each window
    """
    cfg = SUBSYSTEMS[subsystem_type]
    id_col, ts_col, signal_cols = cfg["id_col"], cfg["timestamp_col"], cfg["signal_cols"]

    X_list, end_index_list, unit_list = [], [], []
    for uid in unit_ids:
        sub = df[df[id_col] == uid].sort_values(ts_col)
        if len(sub) < window_length:
            continue
        vals = scaler.transform(sub[signal_cols].values).astype(np.float32)
        idx = sub.index.values
        n = len(sub)
        for start in range(0, n - window_length + 1, stride):
            end = start + window_length
            X_list.append(vals[start:end])
            end_index_list.append(idx[end - 1])
            unit_list.append(uid)

    if not X_list:
        return (np.empty((0, window_length, len(signal_cols)), dtype=np.float32),
                np.array([], dtype=np.int64), np.array([], dtype=object))
    return np.stack(X_list), np.array(end_index_list), np.array(unit_list)


def train_autoencoder(X_train, X_val, n_features, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE,
                       lr=AE_LEARNING_RATE, patience=AE_EARLY_STOP_PATIENCE, verbose=True):
    """
    X_train / X_val must already be disjoint by UNIT (see
    data_loader.split_healthy_units) -- this function does not split
    further, it just trains on X_train and tracks val loss on X_val for
    early stopping / checkpoint selection.
    """
    torch.manual_seed(RANDOM_SEED)
    device = get_device()
    rng = np.random.default_rng(RANDOM_SEED)

    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()

    model = SequenceAutoencoder(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_train_loss = None
    best_state = None
    epochs_no_improve = 0

    n_train = len(X_train_t)
    n_val = len(X_val_t)
    for epoch in range(epochs):
        model.train()
        epoch_perm = torch.from_numpy(rng.permutation(n_train))
        total_loss = 0.0
        for start in range(0, n_train, batch_size):
            batch_idx = epoch_perm[start:start + batch_size]
            batch = X_train_t[batch_idx].to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        train_loss = total_loss / n_train

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                batch = X_val_t[start:start + batch_size].to(device)
                val_recon = model(batch)
                val_total_loss += loss_fn(val_recon, batch).item() * len(batch)
        val_loss = val_total_loss / n_val

        if verbose:
            print(f"    epoch {epoch + 1:02d}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"    early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.5f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, device, best_train_loss, best_val_loss


def score_windows(model, device, X, batch_size=AE_BATCH_SIZE):
    """Per-window mean squared reconstruction error (scalar per window)."""
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).float().to(device)
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=(1, 2))
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors) if errors else np.array([])
