"""Neural-network-based compressor built on a tiny autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .base import BaseCompressor


try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - handled at runtime
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


class _LinearAutoencoder(nn.Module):  # type: ignore[misc]
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(n_features, latent_dim)
        self.decoder = nn.Linear(latent_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised implicitly
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


@dataclass
class NeuralAutoencoderCompressor(BaseCompressor):
    """Compress data with a shallow neural autoencoder."""

    latent_dim: int = 64
    epochs: int = 200
    learning_rate: float = 1e-3
    device: str | None = None
    name: str = "neural_autoencoder"

    def __post_init__(self):
        if torch is None:
            raise ImportError(
                "PyTorch is required for NeuralAutoencoderCompressor"  # pragma: no cover - executed when torch missing
            ) from TORCH_IMPORT_ERROR
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def compress(self, data: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        data = np.asarray(data, dtype=np.float32)
        x = data.reshape(1, -1)
        n_features = x.shape[1]

        latent_dim = int(kwargs.get("latent_dim", self.latent_dim))
        epochs = int(kwargs.get("epochs", self.epochs))
        lr = float(kwargs.get("learning_rate", self.learning_rate))

        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = _LinearAutoencoder(n_features, latent_dim).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        x_tensor = torch.from_numpy(x).to(device)
        mean = x_tensor.mean()
        std = x_tensor.std()
        if float(std) == 0.0:
            std = torch.tensor(1.0, device=device)
        x_norm = (x_tensor - mean) / std

        for _ in range(epochs):
            optimiser.zero_grad(set_to_none=True)
            x_hat = model(x_norm)
            loss = criterion(x_hat, x_norm)
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            latent = model.encoder(x_norm)
            recon = model.decoder(latent)

        latent_np = latent.cpu().numpy().astype(np.float32)
        enc_w = model.encoder.weight.detach().cpu().numpy().astype(np.float32)
        enc_b = model.encoder.bias.detach().cpu().numpy().astype(np.float32)
        dec_w = model.decoder.weight.detach().cpu().numpy().astype(np.float32)
        dec_b = model.decoder.bias.detach().cpu().numpy().astype(np.float32)

        return {
            "shape": np.asarray(data.shape, dtype=np.int32),
            "dtype": np.asarray([data.dtype.str.encode("ascii")], dtype="S16"),
            "latent": latent_np,
            "encoder_weight": enc_w,
            "encoder_bias": enc_b,
            "decoder_weight": dec_w,
            "decoder_bias": dec_b,
            "mean": np.asarray([float(mean.cpu().item())], dtype=np.float32),
            "std": np.asarray([float(std.cpu().item())], dtype=np.float32),
        }

    def decompress(self, compressed_data: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        shape = tuple(int(v) for v in compressed_data["shape"].astype(int))
        dtype_str = compressed_data["dtype"][0].decode("ascii")
        dtype = np.dtype(dtype_str)

        latent = torch.from_numpy(np.asarray(compressed_data["latent"], dtype=np.float32))
        enc_w = torch.from_numpy(np.asarray(compressed_data["encoder_weight"], dtype=np.float32))
        enc_b = torch.from_numpy(np.asarray(compressed_data["encoder_bias"], dtype=np.float32))
        dec_w = torch.from_numpy(np.asarray(compressed_data["decoder_weight"], dtype=np.float32))
        dec_b = torch.from_numpy(np.asarray(compressed_data["decoder_bias"], dtype=np.float32))

        latent_dim = latent.shape[-1]
        n_features = dec_w.shape[-1]

        model = _LinearAutoencoder(n_features, latent_dim)
        model.encoder.weight.data = enc_w
        model.encoder.bias.data = enc_b
        model.decoder.weight.data = dec_w
        model.decoder.bias.data = dec_b

        mean = float(np.asarray(compressed_data["mean"], dtype=np.float32)[0])
        std = float(np.asarray(compressed_data["std"], dtype=np.float32)[0])
        if std == 0:
            std = 1.0

        with torch.no_grad():
            recon = model.decoder(latent)
        recon = recon.numpy().reshape(shape)
        recon = recon * std + mean
        return recon.astype(dtype, copy=False)


__all__ = ["NeuralAutoencoderCompressor"]

