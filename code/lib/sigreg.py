import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg).

    Measures how far a batch of latent vectors deviates from an isotropic
    standard Gaussian N(0, I).  Uses the Epps-Pulley characteristic-function
    statistic on random 1-D projections of the samples.

    The returned value is:
      - Differentiable w.r.t. z (gradients flow through cos/sin and standardisation)
      - Zero iff every 1-D projection of z is exactly N(0, 1)
      - Positive and growing for any departure from Gaussianity or isotropy

    Role in DPersona
    ----------------
    Stage 1 — added to elbo():
        KL(posterior || prior) aligns each image's posterior with its own prior,
        but never enforces a globally well-structured latent space.  SIGReg on
        batched posterior samples fills that gap: it pushes the *aggregate*
        posterior distribution p(z) = ∫ q(z|x) p(x) dx toward N(0, I).

    Stage 2 — added to train_step:
        Applied to the concatenated per-expert projected latents [B*E, D].
        Prevents any expert's projection head from collapsing its outputs to a
        single point or a degenerate subspace.

    Adapted from LeJEPA (Balestriero & LeCun, arXiv:2511.08544).

    Args:
        num_slices:  Number of random 1-D projection directions (more = stronger
                     test power but slightly more compute).
        t_max:       Upper integration limit for the EP characteristic-function test.
        n_points:    Number of quadrature points (must be odd).
        standardize: If True, standardize z per-dimension across the batch before
                     testing.  This makes SIGReg complementary to the KL term:
                     KL handles mean / variance; SIGReg checks Gaussianness of shape.
    """

    def __init__(
        self,
        num_slices: int = 64,
        t_max: float = 3.0,
        n_points: int = 17,
        standardize: bool = True,
    ):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd"
        self.num_slices = num_slices
        self.standardize = standardize

        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2.0 * dt)
        weights[0] = dt
        weights[-1] = dt
        phi = (-0.5 * t.square()).exp()   # real part of N(0,1) characteristic fn
        self.register_buffer("t", t)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("phi", phi)

    def _epps_pulley(self, x: torch.Tensor) -> torch.Tensor:
        """
        Epps-Pulley statistic for a set of 1-D projections.

        x : [N, S]  (N samples across S slices)
        Returns a scalar: mean EP statistic over slices, N-scaled.
        """
        N = x.size(0)
        x_t = x.unsqueeze(-1) * self.t           # [N, S, T]
        cos_mean = x_t.cos().mean(0)              # [S, T]
        sin_mean = x_t.sin().mean(0)              # [S, T]
        err = (cos_mean - self.phi).square() + sin_mean.square()  # [S, T]
        return (err * self.weights).sum(-1).mean() * N            # scalar

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : [N, D]  — a batch of latent vectors.
        Returns a differentiable scalar loss (≥ 0).
        """
        if z.shape[0] < 4:
            return z.new_zeros(())

        if self.standardize:
            z = (z - z.mean(0)) / (z.std(0) + 1e-6)

        D = z.shape[1]
        A = torch.randn(D, self.num_slices, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)   # unit-norm projection directions
        return self._epps_pulley(z @ A)
