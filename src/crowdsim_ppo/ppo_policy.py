"""Small PPO implementation for CrowdSim robot navigation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class RobotPPOConfig:
    obs_dim: int
    action_dim: int = 2
    hidden_dims: tuple[int, ...] = (128,)
    vector_hidden_dims: tuple[int, ...] | None = None
    map_projection_hidden_dims: tuple[int, ...] | None = None
    actor_hidden_dims: tuple[int, ...] | None = None
    critic_hidden_dims: tuple[int, ...] | None = None
    map_encoder_channels: tuple[int, ...] = (8, 16)
    map_encoder_kernel_sizes: tuple[int, ...] = (3, 3)
    map_encoder_strides: tuple[int, ...] = (2, 2)
    map_encoder_paddings: tuple[int, ...] = (1, 1)
    map_feature_dim: int = 64
    vector_obs_dim: int | None = None
    map_size: int = 0
    map_enabled: bool = True
    depth_enabled: bool = True
    depth_size: int = 224
    depth_encoder_channels: tuple[int, ...] = (16, 32)
    depth_encoder_kernel_sizes: tuple[int, ...] = (3, 3)
    depth_encoder_strides: tuple[int, ...] = (2, 2)
    depth_encoder_paddings: tuple[int, ...] = (1, 1)
    depth_feature_dim: int = 64
    attn_enabled: bool = False
    attn_dim: int = 128
    neighbor_dim: int = 5
    num_neighbors: int = 4
    lr: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    minibatch_size: int = 256


class DepthEncoder(nn.Module):
    """Lightweight CNN encoder for 1-channel depth images.

    Convolutional layers downsample the 224×224 input, then a 4×4 adaptive
    pool keeps a coarse spatial grid before flattening.  The
    previous AdaptiveAvgPool2d(1) collapsed the whole feature map to a single
    vector, destroying left/center/right spatial structure — which is the only
    obstacle-direction cue the policy has now that the occupancy map no longer
    enters the observation.  Keeping a 4×4 grid preserves that cue.

    Layout: 224 → conv stack → AdaptiveAvgPool(4,4) → Flatten → Linear → LayerNorm → Tanh

    LayerNorm on the output ensures the depth features start at a comparable
    magnitude to the vec/map features regardless of random initialisation,
    preventing the depth branch from dominating gradients early in training.
    """

    def __init__(
        self,
        depth_size: int = 224,
        channels: tuple[int, ...] = (16, 32),
        kernel_sizes: tuple[int, ...] = (3, 3),
        strides: tuple[int, ...] = (2, 2),
        paddings: tuple[int, ...] = (1, 1),
        feature_dim: int = 64,
        spatial_grid: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, s, p in zip(channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_ch, out_ch, k, s, p))
            layers.append(nn.ReLU())
            in_ch = out_ch
        # Preserve coarse spatial structure (4×4 grid) instead of collapsing
        # to a single vector — left/center/right obstacle cues survive.
        layers.append(nn.AdaptiveAvgPool2d((spatial_grid, spatial_grid)))
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        # Determine flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, depth_size, depth_size)
            flat_dim = int(self.cnn(dummy).shape[-1])
        self.projection     = nn.Linear(flat_dim, feature_dim)
        self.projection_act = nn.Tanh()
        # LayerNorm stabilises output magnitude at init, preventing the depth
        # branch from injecting large-scale noise into the fused feature vector.
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # depth: (N, H, W) or (N, 1, H, W)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)  # (N, 1, H, W)
        x = self.cnn(depth)
        return self.output_norm(self.projection_act(self.projection(x)))


class MapEncoder(nn.Module):
    """CNN encoder for the ego-centric occupancy-map patch (1-channel).

    Mirrors DepthEncoder so map is a pluggable channel consumed via a standalone
    module (not sliced out of the vector obs).  Input: (N, map_size, map_size)
    float where 1.0 = obstacle / unknown, 0.0 = free.  Output: (N, feature_dim).

    Layout: Conv stack → AdaptiveAvgPool(spatial_grid) → Flatten → Linear → Tanh → LayerNorm.
    """

    def __init__(
        self,
        map_size: int = 24,
        channels: tuple[int, ...] = (8, 16),
        kernel_sizes: tuple[int, ...] = (3, 3),
        strides: tuple[int, ...] = (2, 2),
        paddings: tuple[int, ...] = (1, 1),
        feature_dim: int = 64,
        spatial_grid: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, s, p in zip(channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_ch, out_ch, k, s, p))
            layers.append(nn.ReLU())
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((spatial_grid, spatial_grid)))
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, map_size, map_size)
            flat_dim = int(self.cnn(dummy).shape[-1])
        self.projection = nn.Linear(flat_dim, feature_dim)
        self.projection_act = nn.Tanh()
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, map_patch: torch.Tensor) -> torch.Tensor:
        # map_patch: (N, H, W) or (N, 1, H, W)
        if map_patch.dim() == 3:
            map_patch = map_patch.unsqueeze(1)  # (N, 1, H, W)
        x = self.cnn(map_patch)
        return self.output_norm(self.projection_act(self.projection(x)))


def flatten_neighbor_observation(
    obs: torch.Tensor,
    neighbors: torch.Tensor,
    neighbor_mask: torch.Tensor,
) -> torch.Tensor:
    """Rebuild the proven ego + closest + sorted Top-K vector observation.

    Neighbor tensors remain an independent public channel.  They are masked
    and flattened only at the vector-encoder boundary.  ``nav_manager`` emits
    neighbors in ascending-distance order, so index zero is the closest one.
    """
    if neighbors.ndim != 3 or neighbor_mask.shape != neighbors.shape[:2]:
        raise ValueError(
            "Expected neighbors [B,K,D] and neighbor_mask [B,K], got "
            f"{tuple(neighbors.shape)} and {tuple(neighbor_mask.shape)}"
        )
    if obs.shape[0] != neighbors.shape[0]:
        raise ValueError("obs and neighbors must have the same batch size")

    masked = neighbors * neighbor_mask.to(neighbors.dtype).unsqueeze(-1)
    closest = masked[:, 0] if masked.shape[1] else masked.new_zeros((masked.shape[0], 0))
    return torch.cat((obs, closest, masked.flatten(start_dim=1)), dim=-1)


class RGBEncoder(nn.Module):
    """CNN encoder for RGB images (3-channel).

    Mirrors MapEncoder/DepthEncoder so RGB is a pluggable channel.  Input:
    (N, 3, H, W) or (N, H, W, 3); output: (N, feature_dim).  Used by the flow
    policy's CNN-mode RGB branch (ViT mode uses ViTRGBEncoder instead).
    """

    def __init__(
        self,
        rgb_size: int = 224,
        channels: tuple[int, ...] = (32, 64),
        kernel_sizes: tuple[int, ...] = (3, 3),
        strides: tuple[int, ...] = (2, 2),
        paddings: tuple[int, ...] = (1, 1),
        feature_dim: int = 256,
        spatial_grid: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch, k, s, p in zip(channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_ch, out_ch, k, s, p))
            layers.append(nn.ReLU())
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((spatial_grid, spatial_grid)))
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, rgb_size, rgb_size)
            flat_dim = int(self.cnn(dummy).shape[-1])
        self.projection = nn.Linear(flat_dim, feature_dim)
        self.projection_act = nn.Tanh()
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: (N, 3, H, W) expected; accept (N, H, W, 3) by permuting.
        if rgb.dim() == 4 and rgb.shape[-1] == 3:
            rgb = rgb.permute(0, 3, 1, 2).contiguous()
        x = self.cnn(rgb)
        return self.output_norm(self.projection_act(self.projection(x)))


class FusionAttention(nn.Module):
    """Small self-attention over the per-modality encoder outputs.

    Each encoder (vector / map / depth) produces a feature vector of possibly
    different dim.  We project each to ``attn_dim`` so they become a sequence
    of tokens (one per modality), run one Transformer-encoder layer so the
    modalities attend to each other, then flatten the tokens back into a
    single fused vector (num_tokens * attn_dim).  When disabled, the caller
    falls back to a plain ``torch.cat`` of the raw encoder outputs.

    Kept tiny (1 layer, ~num_tokens=3) so it adds negligible compute vs the
    CNN encoders — the goal is cross-modal gating, not a heavy backbone.
    """

    def __init__(
        self,
        in_dims: list[int],
        attn_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_tokens = len(in_dims)
        self.attn_dim = int(attn_dim)
        # Project each modality's feature into the shared token dim.
        self.projections = nn.ModuleList(
            [nn.Linear(d, attn_dim) for d in in_dims]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.out_dim = self.num_tokens * attn_dim

    def forward(self, feature_list: list[torch.Tensor]) -> torch.Tensor:
        # feature_list: each (B, d_i) → project → (B, num_tokens, attn_dim)
        tokens = torch.stack([proj(f) for proj, f in zip(self.projections, feature_list)], dim=1)
        tokens = self.transformer(tokens)            # (B, num_tokens, attn_dim)
        return tokens.flatten(1)                      # (B, num_tokens * attn_dim)


class RobotActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (128,),
        vector_hidden_dims: tuple[int, ...] | None = None,
        map_projection_hidden_dims: tuple[int, ...] | None = None,
        actor_hidden_dims: tuple[int, ...] | None = None,
        critic_hidden_dims: tuple[int, ...] | None = None,
        map_encoder_channels: tuple[int, ...] = (8, 16),
        map_encoder_kernel_sizes: tuple[int, ...] = (3, 3),
        map_encoder_strides: tuple[int, ...] = (2, 2),
        map_encoder_paddings: tuple[int, ...] = (1, 1),
        map_feature_dim: int = 64,
        vector_obs_dim: int | None = None,
        map_size: int = 0,
        map_enabled: bool = True,
        depth_enabled: bool = True,
        depth_size: int = 224,
        depth_encoder_channels: tuple[int, ...] = (16, 32),
        depth_encoder_kernel_sizes: tuple[int, ...] = (3, 3),
        depth_encoder_strides: tuple[int, ...] = (2, 2),
        depth_encoder_paddings: tuple[int, ...] = (1, 1),
        depth_feature_dim: int = 64,
        attn_enabled: bool = False,
        attn_dim: int = 128,
        neighbor_dim: int = 5,
        num_neighbors: int = 4,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.vector_obs_dim = int(vector_obs_dim or obs_dim)
        self.map_size = int(map_size)
        self.map_obs_dim = self.map_size * self.map_size
        self.map_enabled = bool(map_enabled) and self.map_size > 0
        self.depth_enabled = bool(depth_enabled)
        self.attn_enabled = bool(attn_enabled)
        self.num_neighbors = int(num_neighbors)
        self.neighbor_dim = int(neighbor_dim)
        hidden_dims = normalize_dims(hidden_dims)
        vector_hidden_dims = normalize_dims(vector_hidden_dims or hidden_dims)
        actor_hidden_dims = normalize_dims(actor_hidden_dims or hidden_dims)
        critic_hidden_dims = normalize_dims(critic_hidden_dims or hidden_dims)
        self.has_map = self.map_enabled
        self.vector_encoder = make_mlp(
            input_dim=self.vector_obs_dim + (self.num_neighbors + 1) * self.neighbor_dim,
            hidden_dims=vector_hidden_dims,
        )
        # Track each encoder's output dim for the optional fusion attention.
        encoder_dims = [vector_hidden_dims[-1]]
        if self.has_map:
            self.map_encoder = MapEncoder(
                map_size=self.map_size,
                channels=map_encoder_channels,
                kernel_sizes=map_encoder_kernel_sizes,
                strides=map_encoder_strides,
                paddings=map_encoder_paddings,
                feature_dim=map_feature_dim,
            )
            encoder_dims.append(map_feature_dim)
        else:
            self.map_encoder = None

        if self.depth_enabled:
            self.depth_size = depth_size
            self.depth_encoder = DepthEncoder(
                depth_size=depth_size,
                channels=depth_encoder_channels,
                kernel_sizes=depth_encoder_kernel_sizes,
                strides=depth_encoder_strides,
                paddings=depth_encoder_paddings,
                feature_dim=depth_feature_dim,
            )
            encoder_dims.append(depth_feature_dim)
        else:
            self.depth_size = 0
            self.depth_encoder = None

        # Optional cross-modal self-attention over encoder outputs.  Replaces
        # the plain cat with token-attention fusion; out_dim = num_tokens * attn_dim.
        if self.attn_enabled:
            self.fusion = FusionAttention(in_dims=encoder_dims, attn_dim=attn_dim)
            feature_dim = self.fusion.out_dim
        else:
            self.fusion = None
            feature_dim = sum(encoder_dims)

        self.actor = make_mlp(
            input_dim=feature_dim,
            hidden_dims=actor_hidden_dims,
            output_dim=action_dim,
        )
        self.critic = make_mlp(
            input_dim=feature_dim,
            hidden_dims=critic_hidden_dims,
            output_dim=1,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Start from a neutral raw command mean. Exploration still comes from
        # the Gaussian standard deviation.
        if action_dim >= 1:
            with torch.no_grad():
                # Zero output head gives raw mean [0, 0] for every initial
                # observation: sigmoid(0)=0.5 speed and tanh(0)=0 yaw rate.
                self.actor[-1].weight.zero_()
                self.actor[-1].bias.zero_()

    def encode(
        self, obs: torch.Tensor, depth: torch.Tensor | None = None,
        map_patch: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if neighbors is None or neighbor_mask is None:
            raise ValueError("neighbors and neighbor_mask are required")
        vector_obs = flatten_neighbor_observation(
            obs[:, : self.vector_obs_dim], neighbors, neighbor_mask
        )
        features = [self.vector_encoder(vector_obs)]
        if self.has_map:
            # map is a standalone channel (like depth) — not sliced from obs.
            # Guard against None / empty-batch the same way depth does.
            if map_patch is None or map_patch.shape[0] == 0:
                map_patch = torch.zeros(obs.shape[0], self.map_size, self.map_size, device=obs.device)
            features.append(self.map_encoder(map_patch))
        if self.depth_enabled:
            # Guard against both None and empty-batch depth (a (0, S, S) tensor
            # would flow through the depth encoder and produce a (0, F) feature
            # that fails to cat with the (N, F) vector feature).  Treat either
            # case as "no depth available" and fall back to a zero filling that
            # matches the current obs batch size.
            if depth is None or depth.shape[0] == 0:
                depth = torch.zeros(obs.shape[0], self.depth_size, self.depth_size, device=obs.device)
            features.append(self.depth_encoder(depth))
        if self.fusion is not None:
            return self.fusion(features)
        return torch.cat(features, dim=-1) if len(features) > 1 else features[0]

    def forward(
        self, obs: torch.Tensor, depth: torch.Tensor | None = None,
        map_patch: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(obs, depth, map_patch, neighbors, neighbor_mask)
        mean = self.actor(features)
        value = self.critic(features).squeeze(-1)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std, value

    def act(
        self, obs: torch.Tensor, depth: torch.Tensor | None = None,
        map_patch: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs, depth, map_patch, neighbors, neighbor_mask)
        dist = Normal(mean, log_std.exp())
        raw_action = dist.rsample()
        action = bounded_robot_action(raw_action)
        log_prob = bounded_robot_action_log_prob(dist, raw_action, action)
        return action, raw_action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        raw_actions: torch.Tensor,
        depth: torch.Tensor | None = None,
        map_patch: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs, depth, map_patch, neighbors, neighbor_mask)
        dist = Normal(mean, log_std.exp())
        actions = bounded_robot_action(raw_actions)
        log_prob = bounded_robot_action_log_prob(dist, raw_actions, actions)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


def make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = int(input_dim)
    for hidden_dim in normalize_dims(hidden_dims):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.Tanh())
        current_dim = hidden_dim
    if output_dim is not None:
        layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def make_conv_encoder(
    channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    strides: tuple[int, ...],
    paddings: tuple[int, ...],
) -> nn.Sequential:
    channels = normalize_dims(channels)
    kernel_sizes = expand_or_validate(kernel_sizes, len(channels), "kernel_sizes")
    strides = expand_or_validate(strides, len(channels), "strides")
    paddings = expand_or_validate(paddings, len(channels), "paddings")
    layers: list[nn.Module] = []
    in_channels = 1
    for out_channels, kernel_size, stride, padding in zip(
        channels,
        kernel_sizes,
        strides,
        paddings,
    ):
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


def normalize_dims(value: int | float | str | list | tuple) -> tuple[int, ...]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        dims = tuple(int(dim) for dim in value)
    else:
        dims = (int(value),)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"Network dimensions must be positive, got {value}.")
    return dims


def expand_or_validate(value: int | float | str | list | tuple, length: int, name: str) -> tuple[int, ...]:
    values = normalize_dims(value)
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(f"Expected {name} to have length 1 or {length}, got {values}.")
    return values


def robot_network_kwargs(
    network_cfg: dict,
    hidden_dim_override: int | None = None,
    num_layers_override: int | None = None,
    depth_enabled: bool | None = None,
    depth_size: int | None = None,
    map_enabled: bool | None = None,
    map_size: int | None = None,
    num_neighbors: int | None = None,
) -> dict:
    """Build keyword arguments for RobotActorCritic from config dicts.

    ``depth_enabled``/``depth_size`` and ``map_enabled``/``map_size`` are the
    authoritative overrides — callers should pass these from
    ``nav_manager.config``.  When omitted they fall back to conservative
    defaults (depth: True/224, map: True/0).
    """
    num_layers = int(num_layers_override or network_cfg.get("num_layers", 1))
    shared_value = (
        hidden_dim_override
        if hidden_dim_override is not None
        else network_cfg.get("hidden_dims", network_cfg.get("hidden_dim", 128))
    )
    shared_dims = network_hidden_dims(shared_value, num_layers)
    map_encoder_cfg = network_cfg.get("map_encoder", {})
    if not isinstance(map_encoder_cfg, dict):
        map_encoder_cfg = {}
    depth_cfg = network_cfg.get("depth_encoder", {})
    if not isinstance(depth_cfg, dict):
        depth_cfg = {}
    kwargs = {
        "hidden_dims": shared_dims,
        "vector_hidden_dims": optional_hidden_dims(
            network_cfg.get("vector_hidden_dims"),
            default=shared_dims,
        ),
        "map_projection_hidden_dims": optional_hidden_dims(
            network_cfg.get("map_projection_hidden_dims"),
            default=shared_dims,
        ),
        "actor_hidden_dims": optional_hidden_dims(
            network_cfg.get("actor_hidden_dims"),
            default=shared_dims,
        ),
        "critic_hidden_dims": optional_hidden_dims(
            network_cfg.get("critic_hidden_dims"),
            default=shared_dims,
        ),
        "map_encoder_channels": normalize_dims(map_encoder_cfg.get("channels", (8, 16))),
        "map_encoder_kernel_sizes": normalize_dims(map_encoder_cfg.get("kernel_sizes", (3, 3))),
        "map_encoder_strides": normalize_dims(map_encoder_cfg.get("strides", (2, 2))),
        "map_encoder_paddings": normalize_dims(map_encoder_cfg.get("paddings", (1, 1))),
    }
    # depth: explicit caller param > default True/224
    kwargs["depth_enabled"] = bool(depth_enabled) if depth_enabled is not None else True
    kwargs["depth_size"] = int(depth_size) if depth_size is not None else 224

    # map: explicit caller param > default True/0 (0 = disabled)
    kwargs["map_enabled"] = bool(map_enabled) if map_enabled is not None else True
    kwargs["map_size"] = int(map_size) if map_size is not None else 0
    kwargs["map_feature_dim"] = int(map_encoder_cfg.get("feature_dim", 64))

    # fusion attention: optional cross-modal self-attention over encoder
    # outputs.  Off by default; enable via network.attention.enabled in yaml.
    attn_cfg = network_cfg.get("attention", {})
    if not isinstance(attn_cfg, dict):
        attn_cfg = {}
    kwargs["attn_enabled"] = bool(attn_cfg.get("enabled", False))
    kwargs["attn_dim"] = int(attn_cfg.get("dim", 128))

    kwargs["neighbor_dim"] = 5
    kwargs["num_neighbors"] = int(num_neighbors if num_neighbors is not None else 4)

    kwargs["depth_encoder_channels"] = normalize_dims(depth_cfg.get("channels", (16, 32, 64, 128)))
    kwargs["depth_encoder_kernel_sizes"] = normalize_dims(depth_cfg.get("kernel_sizes", (3, 3, 3, 3)))
    kwargs["depth_encoder_strides"] = normalize_dims(depth_cfg.get("strides", (2, 2, 2, 2)))
    kwargs["depth_encoder_paddings"] = normalize_dims(depth_cfg.get("paddings", (1, 1, 1, 1)))
    kwargs["depth_feature_dim"] = int(depth_cfg.get("feature_dim", 256))
    return kwargs


def network_hidden_dims(value, num_layers: int) -> tuple[int, ...]:
    dims = normalize_dims(value)
    if len(dims) == 1:
        return dims * max(1, int(num_layers))
    return dims


def optional_hidden_dims(value, default: tuple[int, ...]) -> tuple[int, ...] | None:
    if value is None:
        return None
    return normalize_dims(value)


def bounded_robot_action(raw_action: torch.Tensor) -> torch.Tensor:
    """Apply tanh to both latents, then affine-map linear speed to (0, 1)."""
    if raw_action.shape[-1] != 2:
        raise ValueError(f"Expected [..., 2] robot action, got {tuple(raw_action.shape)}")
    squashed = torch.tanh(raw_action)
    return torch.stack(
        (0.5 * (squashed[..., 0] + 1.0), squashed[..., 1]),
        dim=-1,
    )


def bounded_robot_action_log_prob(
    dist: Normal,
    raw_action: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """Exact density for affine-tanh linear and tanh angular actions."""
    log_prob = dist.log_prob(raw_action).sum(dim=-1)
    del action  # Jacobians are evaluated stably from the pre-transform latent.
    raw_linear = raw_action[..., 0]
    raw_angular = raw_action[..., 1]
    linear_jacobian = -math.log(2.0) + 2.0 * (
        math.log(2.0) - raw_linear - F.softplus(-2.0 * raw_linear)
    )
    angular_jacobian = 2.0 * (
        math.log(2.0) - raw_angular - F.softplus(-2.0 * raw_angular)
    )
    return log_prob - linear_jacobian - angular_jacobian


class RobotRolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        depth_size: int = 0,
        map_size: int = 0,
        num_neighbors: int = 0,
        neighbor_dim: int = 5,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros(rollout_steps, num_envs, obs_dim, device=device)
        self.has_neighbors = num_neighbors > 0
        if self.has_neighbors:
            self.neighbors = torch.zeros(
                rollout_steps, num_envs, num_neighbors, neighbor_dim, device=device
            )
            self.neighbor_mask = torch.zeros(
                rollout_steps, num_envs, num_neighbors, dtype=torch.bool, device=device
            )
        self.has_depth = depth_size > 0
        if self.has_depth:
            self.depth = torch.zeros(rollout_steps, num_envs, depth_size, depth_size, device=device)
        self.has_map = map_size > 0
        if self.has_map:
            self.map = torch.zeros(rollout_steps, num_envs, map_size, map_size, device=device)
        self.raw_actions = torch.zeros(rollout_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(rollout_steps, num_envs, device=device)
        self.rewards = torch.zeros(rollout_steps, num_envs, device=device)
        self.dones = torch.zeros(rollout_steps, num_envs, device=device)
        self.values = torch.zeros(rollout_steps, num_envs, device=device)
        self.advantages = torch.zeros(rollout_steps, num_envs, device=device)
        self.returns = torch.zeros(rollout_steps, num_envs, device=device)
        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        raw_actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        depth: torch.Tensor | None = None,
        map_patch: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> None:
        if self.step >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer is full.")
        self.obs[self.step].copy_(obs)
        if self.has_neighbors:
            if neighbors is not None:
                self.neighbors[self.step].copy_(neighbors)
            if neighbor_mask is not None:
                self.neighbor_mask[self.step].copy_(neighbor_mask)
        if self.has_depth:
            if depth is not None:
                self.depth[self.step].copy_(depth)
            else:
                # Buffer slot keeps its zero-initialised value.
                # Warn once so that silent fallback is visible in logs.
                if not getattr(self, "_warned_null_depth", False):
                    import logging
                    logging.getLogger("CrowdSim").warning(
                        "RolloutBuffer.add: depth=None at step %d while has_depth=True — "
                        "this buffer slot will be all-zeros.  Check camera / depth pipeline.",
                        self.step,
                    )
                    self._warned_null_depth = True
        if self.has_map:
            if map_patch is not None:
                self.map[self.step].copy_(map_patch)
            # else: keep zero-initialised slot (map is always available when
            # enabled, so None here is unexpected — silent zero fill is fine).
        self.raw_actions[self.step].copy_(raw_actions)
        self.log_probs[self.step].copy_(log_probs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones.float())
        self.values[self.step].copy_(values)
        self.step += 1

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        advantage = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.rollout_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[t] = advantage
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std(unbiased=False) + 1.0e-8
        )

    def batches(self, minibatch_size: int):
        total = self.rollout_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)
        flat = {
            "obs": self.obs.reshape(total, -1),
            "raw_actions": self.raw_actions.reshape(total, -1),
            "log_probs": self.log_probs.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
        }
        if self.has_depth:
            flat["depth"] = self.depth.reshape(total, self.depth.shape[-2], self.depth.shape[-1])
        if self.has_map:
            flat["map"] = self.map.reshape(total, self.map.shape[-2], self.map.shape[-1])
        if self.has_neighbors:
            flat["neighbors"] = self.neighbors.reshape(total, *self.neighbors.shape[2:])
            flat["neighbor_mask"] = self.neighbor_mask.reshape(total, self.neighbor_mask.shape[-1])
        for start in range(0, total, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            yield {key: value[batch_idx] for key, value in flat.items()}

    def reset(self) -> None:
        self.step = 0


class RobotPPOTrainer:
    def __init__(self, config: RobotPPOConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.model = RobotActorCritic(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            vector_hidden_dims=config.vector_hidden_dims,
            map_projection_hidden_dims=config.map_projection_hidden_dims,
            actor_hidden_dims=config.actor_hidden_dims,
            critic_hidden_dims=config.critic_hidden_dims,
            map_encoder_channels=config.map_encoder_channels,
            map_encoder_kernel_sizes=config.map_encoder_kernel_sizes,
            map_encoder_strides=config.map_encoder_strides,
            map_encoder_paddings=config.map_encoder_paddings,
            map_feature_dim=config.map_feature_dim,
            vector_obs_dim=config.vector_obs_dim,
            map_size=config.map_size,
            map_enabled=config.map_enabled,
            depth_enabled=config.depth_enabled,
            depth_size=config.depth_size,
            depth_encoder_channels=config.depth_encoder_channels,
            depth_encoder_kernel_sizes=config.depth_encoder_kernel_sizes,
            depth_encoder_strides=config.depth_encoder_strides,
            depth_encoder_paddings=config.depth_encoder_paddings,
            depth_feature_dim=config.depth_feature_dim,
            attn_enabled=config.attn_enabled,
            attn_dim=config.attn_dim,
            neighbor_dim=config.neighbor_dim,
            num_neighbors=config.num_neighbors,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.extra_state: dict = {}

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, depth: torch.Tensor | None = None,
        map_patch: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.act(obs, depth, map_patch, neighbors, neighbor_mask)

    @torch.no_grad()
    def value(
        self, obs: torch.Tensor, depth: torch.Tensor | None = None,
        map_patch: torch.Tensor | None = None,
        neighbors: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(obs, depth, map_patch, neighbors, neighbor_mask)[2]

    def update(self, buffer: RobotRolloutBuffer) -> dict[str, float]:
        cfg = self.config
        stats = {
            key: 0.0
            for key in (
                "policy_loss", "value_loss", "entropy", "ratio_mean",
                "ratio_std", "clip_fraction", "approx_kl", "grad_norm",
                "updates",
            )
        }
        return_variance = buffer.returns.var(unbiased=False)
        residual_variance = (buffer.returns - buffer.values).var(unbiased=False)
        explained_variance = 1.0 - residual_variance / (
            return_variance + 1.0e-8
        )
        for _ in range(cfg.ppo_epochs):
            for batch in buffer.batches(cfg.minibatch_size):
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    batch["obs"], batch["raw_actions"], batch.get("depth"), batch.get("map"),
                    batch.get("neighbors"), batch.get("neighbor_mask"),
                )
                log_ratio = new_log_probs - batch["log_probs"]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * batch["advantages"]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                policy_loss = -torch.min(unclipped, clipped * batch["advantages"]).mean()
                # Normalize value loss by return variance so the value
                # function target scale is independent of reward magnitude.
                returns = batch["returns"]
                ret_var = returns.var() + 1e-8
                value_loss = F.mse_loss(values, returns) / ret_var
                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm
                )
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.detach().cpu())
                stats["value_loss"] += float(value_loss.detach().cpu())
                stats["entropy"] += float(entropy_loss.detach().cpu())
                stats["ratio_mean"] += float(ratio.mean().detach().cpu())
                stats["ratio_std"] += float(
                    ratio.std(unbiased=False).detach().cpu()
                )
                stats["clip_fraction"] += float(
                    ((ratio - 1.0).abs() > cfg.clip_ratio)
                    .float().mean().detach().cpu()
                )
                stats["approx_kl"] += float(
                    ((ratio - 1.0) - log_ratio).mean().detach().cpu()
                )
                stats["grad_norm"] += float(grad_norm.detach().cpu())
                stats["updates"] += 1

        count = max(stats["updates"], 1)
        return {
            "policy_loss": stats["policy_loss"] / count,
            "value_loss": stats["value_loss"] / count,
            "entropy": stats["entropy"] / count,
            "ratio_mean": stats["ratio_mean"] / count,
            "ratio_std": stats["ratio_std"] / count,
            "clip_fraction": stats["clip_fraction"] / count,
            "approx_kl": stats["approx_kl"] / count,
            "grad_norm": stats["grad_norm"] / count,
            "explained_variance": float(explained_variance.detach().cpu()),
            "advantage_mean": float(buffer.advantages.mean().detach().cpu()),
            "advantage_std": float(
                buffer.advantages.std(unbiased=False).detach().cpu()
            ),
            "return_mean": float(buffer.returns.mean().detach().cpu()),
            "return_std": float(
                buffer.returns.std(unbiased=False).detach().cpu()
            ),
            "value_mean": float(buffer.values.mean().detach().cpu()),
            "value_std": float(
                buffer.values.std(unbiased=False).detach().cpu()
            ),
            "latent_std_linear": float(
                self.model.log_std[0].exp().detach().cpu()
            ),
            "latent_std_angular": float(
                self.model.log_std[1].exp().detach().cpu()
            ),
        }

    def update_with_kl(
        self,
        buffer: RobotRolloutBuffer,
        ref_model: "RobotActorCritic | None" = None,
        kl_beta: float = 0.0,
    ) -> dict[str, float]:
        """PPO update with optional KL-divergence regularisation toward a reference policy.

        Unlike the original ``update()``, this method computes **true policy KL divergence**
        (not L2 parameter distance) *inside* the optimisation loop, so the KL penalty
        actually affects the gradient.

        KL formula (diagonal Gaussian, action dim *d*):

        .. math::

            D_{KL}(\\pi_{\\text{ref}} \\| \\pi_\\theta)
            = \\sum_{j=1}^{d} \\Bigl[
                \\log \\frac{\\sigma_{\\theta,j}}{\\sigma_{\\text{ref},j}}
                + \\frac{\\sigma_{\\text{ref},j}^2 + (\\mu_{\\text{ref},j} - \\mu_{\\theta,j})^2}
                         {2\\,\\sigma_{\\theta,j}^2}
                - \\tfrac{1}{2}
              \\Bigr]

        The expectation is approximated by the current mini-batch.

        Args:
            buffer:    Rollout buffer (same as ``update()``).
            ref_model: Frozen reference policy (deepcopy of base checkpoint).
                       If ``None`` or ``kl_beta == 0``, behaviour is identical
                       to ``update()``.
            kl_beta:   Weight of the KL penalty term in the total loss.
        """
        cfg = self.config
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_loss": 0.0,
            "updates": 0,
        }
        use_kl = kl_beta > 0.0 and ref_model is not None

        for _ in range(cfg.ppo_epochs):
            for batch in buffer.batches(cfg.minibatch_size):
                obs   = batch["obs"]
                depth = batch.get("depth")
                map_patch = batch.get("map")

                # ── 标准 PPO 损失 ─────────────────────────────────────
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    obs, batch["raw_actions"], depth, map_patch,
                    batch.get("neighbors"), batch.get("neighbor_mask"),
                )
                ratio     = torch.exp(new_log_probs - batch["log_probs"])
                adv       = batch["advantages"]
                unclipped = ratio * adv
                clipped   = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                policy_loss = -torch.min(unclipped, clipped * adv).mean()

                returns   = batch["returns"]
                ret_var   = returns.var() + 1e-8
                value_loss = F.mse_loss(values, returns) / ret_var

                entropy_loss = entropy.mean()

                # ── KL 正则化（解析公式，在循环内与 PPO loss 联合）─────
                kl_term = torch.tensor(0.0, device=self.device)
                if use_kl:
                    # 参考策略：冻结，只做前向传播（无梯度）
                    with torch.no_grad():
                        mu_ref, log_std_ref, _ = ref_model(
                            obs, depth, map_patch, batch.get("neighbors"), batch.get("neighbor_mask")
                        )

                    # 当前策略：参与梯度计算
                    mu_cur, log_std_cur, _ = self.model(
                        obs, depth, map_patch, batch.get("neighbors"), batch.get("neighbor_mask")
                    )

                    # D_KL(π_ref || π_cur) —— 逐动作维度求和，对 batch 取均值
                    # 每项含义：
                    #   (log_std_cur - log_std_ref) → log(σ_cur / σ_ref)
                    #   分子 / (2σ_cur²)           → (σ_ref² + (μ_ref-μ_cur)²) / 2σ_cur²
                    #   - 0.5                       → 常数项
                    std_ref = log_std_ref.exp()
                    # H7: clamp std_cur away from zero to prevent div-by-zero
                    # and NaN gradients when the policy std collapses.
                    std_cur = log_std_cur.exp().clamp(min=1e-6)
                    kl_per_dim = (
                        (log_std_cur - log_std_ref)
                        + (std_ref ** 2 + (mu_ref - mu_cur) ** 2) / (2.0 * std_cur ** 2)
                        - 0.5
                    )
                    kl_term = kl_per_dim.sum(dim=-1).mean()

                # Joint loss (KL is connected to the computation graph).
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_loss
                    + kl_beta * kl_term
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # H7: skip the update step if any gradient is NaN (non-recoverable
                # training crash otherwise).
                if any(p.grad is not None and p.grad.isnan().any()
                       for p in self.model.parameters()):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.detach().cpu())
                stats["value_loss"]  += float(value_loss.detach().cpu())
                stats["entropy"]     += float(entropy_loss.detach().cpu())
                stats["kl_loss"]     += float(kl_term.detach().cpu())
                stats["updates"]     += 1

        count = max(stats["updates"], 1)
        return {
            "policy_loss": stats["policy_loss"] / count,
            "value_loss":  stats["value_loss"]  / count,
            "entropy":     stats["entropy"]      / count,
            "kl_loss":     stats["kl_loss"]      / count,
        }

    def save(self, path: Path, step: int, extra_state: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 3,
                "step": step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "architecture": {
                    "action_parameterization": "affine_tanh_linear_tanh_angular",
                    "num_neighbors": self.model.num_neighbors,
                    "neighbor_dim": self.model.neighbor_dim,
                    "vector_input_dim": self.model.vector_encoder[0].in_features,
                },
                "extra_state": extra_state or {},
            },
            path,
        )

    def load(self, path: Path) -> int:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if int(payload.get("schema_version", 1)) < 3:
            raise ValueError(
                f"Checkpoint {path} predates affine-tanh bounded actions."
            )
        parameterization = payload.get("architecture", {}).get("action_parameterization")
        if parameterization != "affine_tanh_linear_tanh_angular":
            raise ValueError(
                f"Checkpoint {path} has action_parameterization={parameterization!r}; "
                "expected 'affine_tanh_linear_tanh_angular'."
            )
        expected = self.model.vector_encoder[0].in_features
        actual = payload.get("architecture", {}).get("vector_input_dim")
        if actual != expected:
            raise ValueError(
                f"Checkpoint {path} has vector_input_dim={actual}, expected {expected}."
            )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.extra_state = dict(payload.get("extra_state", {}))
        return int(payload.get("step", 0))
