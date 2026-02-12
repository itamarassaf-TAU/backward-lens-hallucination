import torch
import torch.nn as nn

class HallucinationMLP(nn.Module):
    """Small MLP that maps BL features -> similarity score."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
) -> dict:
    """Train MLP classifier with BCE loss on hard labels."""
    
    # REPRODUCIBILITY: Fix the random seed so results are stable
    torch.manual_seed(42)
    
    model = HallucinationMLP(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    train_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # We don't need to print every epoch for such a small model
    print(f"[MLP] Final Train BCE: {train_losses[-1]:.4f}")

    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        val_probs = torch.sigmoid(val_logits)
        val_loss = loss_fn(val_logits, y_val).item()

    print(f"[MLP] Validation BCE: {val_loss:.4f}")

    return {
        "model": model,
        "val_bce": val_loss,
        "val_probs": val_probs.detach().cpu(),
        "train_losses": train_losses,
    }