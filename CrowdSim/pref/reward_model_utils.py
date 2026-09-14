"""Reward model — adapted from pref_nav for CrowdSim."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RewardModel:
    def __init__(self, lr=1e-4, device="cuda"):
        self.device = device

        # ---------- system feature definition ----------
        # state(14) + action(2)
        self.system_dim = 4 + 2*5 + 2

        # ---------- reward net ----------
        self.reward_net = RewardNet_Traj_Multimodal_CrossAttn(
            system_dim=self.system_dim,
        ).to(device)

        self.opt = torch.optim.Adam(self.reward_net.parameters(), lr=lr)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No reward model found at {path}")
            
        state_dict = torch.load(path, map_location=self.device)
        
        self.reward_net.load_state_dict(state_dict)
        self.reward_net.eval()
        
        print(f"\033[92m[RewardModel] Successfully loaded weights from {path}\033[0m")

    # =================================================
    # Encoding (merged StateFeatureEncoder)
    # =================================================
    def encode_step(self, step):
        state  = np.asarray(step["sys_state"], dtype=np.float32)   # (4+2*5)
        action = np.asarray(step["action"], dtype=np.float32) # (2,)

        feat = np.concatenate(
            [
                state,
                action
            ],
            axis=0
        )
        return feat

    # =================================================
    # Segment reward
    # =================================================

    # 输入一段segment，输出奖励和，用于模型evaluation
    @torch.no_grad()
    def sum_reward_from_net(self, segment):

        system_feats = []
        rgb_feats = []
        depth_feats = []

        for step in segment["steps"]:
            system_feats.append(self.encode_step(step))
            rgb_feats.append(step["rgb_feat"])
            depth_feats.append(step["depth"])

        system_feats = torch.tensor(
            np.stack(system_feats),
            device=self.device,
            dtype=torch.float32,
        )

        rgb_feats = torch.tensor(
            np.stack(rgb_feats),
            device=self.device,
            dtype=torch.float32,
        )

        depth_feats = torch.tensor(
            np.stack(depth_feats),
            device=self.device,
            dtype=torch.float32,
        )

        step_rewards = self.reward_net(system_feats, rgb_feats, depth_feats)
        return step_rewards.sum(dim=1)

    @torch.no_grad()
    # 输入一段segment，输出最后一帧奖励，用于rl中推理
    def infer_last_reward(self, segment):

        was_training = self.reward_net.training
        self.reward_net.eval()

        system_feats = []
        rgb_feats = []
        depth_feats = []

        for step in segment["steps"]:
            system_feats.append(self.encode_step(step))
            rgb_feats.append(step["rgb_feat"])
            depth_feats.append(step["depth"])

        system_feats = torch.tensor(
            np.stack(system_feats),
            device=self.device,
            dtype=torch.float32,
        )

        rgb_feats = torch.tensor(
            np.stack(rgb_feats),
            device=self.device,
            dtype=torch.float32,
        )

        depth_feats = torch.tensor(
            np.stack(depth_feats),
            device=self.device,
            dtype=torch.float32,
        )

        step_rewards = self.reward_net(system_feats, rgb_feats, depth_feats)

        result = step_rewards[0, -1].item()

        if was_training:
            self.reward_net.train()

        return result
    
    # prepare batch
    def _prepare_batch(self, segments):
        sys_list, rgb_list, dep_list = [], [], []
        
        for seg in segments:
            steps = seg["steps"]
            sys_list.append([self.encode_step(st) for st in steps])
            rgb_list.append([st["rgb_feat"] for st in steps])
            dep_list.append([st["depth"] for st in steps])

        sys_batch = torch.as_tensor(np.array(sys_list), device=self.device, dtype=torch.float32)
        rgb_batch = torch.as_tensor(np.array(rgb_list), device=self.device, dtype=torch.float32)
        depth_batch = torch.as_tensor(np.array(dep_list), device=self.device, dtype=torch.float32)

        return sys_batch, rgb_batch, depth_batch

    # batch train
    def train_step_batch(self, pairs_batch):
        s1_list = [p[0] for p in pairs_batch]
        s2_list = [p[1] for p in pairs_batch]
        labels = torch.tensor([p[2] for p in pairs_batch], device=self.device).float()

        sys1, rgb1, dep1 = self._prepare_batch(s1_list)
        r1_list = self.reward_net(sys1, rgb1, dep1) # (B, T)
        R1 = r1_list.sum(dim=1) # B

        sys2, rgb2, dep2 = self._prepare_batch(s2_list)
        r2_list = self.reward_net(sys2, rgb2, dep2) # (B, T)
        R2 = r2_list.sum(dim=1) # B

        loss = self.preference_loss(R1, R2, labels, r1_list, r2_list)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 1.0)
        self.opt.step()

        acc = ((R1 > R2).float() == labels).float().mean()
        return float(loss.item()), float(acc.item())

    # Preference loss
    def preference_loss(self, r1, r2, label, r1_list, r2_list):

        # preference ranking loss
        logits = r1 - r2
        pref_loss = F.binary_cross_entropy_with_logits(logits, label)

        # reward magnitude regularization
        l2_reg = 0.01 * (r1**2 + r2**2).mean()

        # -----------------------------
        # baseline = 0 regularization
        # -----------------------------
        zero_reg = 0.01 * (
            r1_list.mean()**2 +
            r2_list.mean()**2
        )

        return pref_loss + l2_reg + zero_reg

    # 模型evaluation
    @torch.no_grad()
    def evaluate_on_world_groups(self, world_groups):
        """
        Evaluate reward model accuracy on labeled world_groups
        """

        self.reward_net.eval()

        correct = 0
        total = 0

        for wid, group in world_groups.items():

            label = group.get("label", None)

            if label is None or label == -1:
                continue

            s1 = group.get("s1", None)
            s2 = group.get("s2", None)

            if s1 is None or s2 is None:
                continue

            R1 = self.sum_reward_from_net(s1)
            R2 = self.sum_reward_from_net(s2)

            pred = int(R1 > R2)

            if pred == label:
                correct += 1

            total += 1

        acc = correct / total if total > 0 else 0.0

        self.reward_net.train()

        print("\n===== RewardNet Full Evaluation =====")
        print(f"Total labeled pairs: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {acc:.4f}")
        print("=====================================\n")

        return acc
    
class RewardNet_Traj_Multimodal_CrossAttn(nn.Module):

    """
    输入：
        system_feats: (B, T, 16)
            = [state(14), action(2)]
        rgb_feats:    (B, T, 384, 16, 16)
        depth_feats:  (B, T, 3, 36, 64)
    输出：
        step_rewards: (B, T)
    """

    def __init__(
        self,
        system_dim=16,
        state_dim=14,
        action_dim=2,
        rgb_embed_dim=64,
        depth_embed_dim=64,
        system_embed_dim=32,
        action_embed_dim=32,
        max_len=200,
    ):
        super().__init__()

        # ================= State Encoder =================
        self.system_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, system_embed_dim),
            nn.LayerNorm(system_embed_dim),
            nn.ReLU(inplace=True)
        )
        # ================= Action Encoder =================
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.ReLU(inplace=True)
        )
        # ================= RGB Encoder =================
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, rgb_embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))
        )
        # ================= Depth Encoder =================
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, depth_embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))
        )
        # ================= Vision projection =================
        self.fusion_dim = 256

        self.state_proj = nn.Linear(system_embed_dim, self.fusion_dim)
        self.action_proj = nn.Linear(action_embed_dim, self.fusion_dim)
        self.vision_proj = nn.Linear(rgb_embed_dim, self.fusion_dim)

        # ================= Positional Encoding =================
        self.state_pos_emb = nn.Parameter(torch.randn(1, max_len, 1, self.fusion_dim))
        self.vision_pos_emb = nn.Parameter(torch.randn(1, max_len, 32, self.fusion_dim))
        self.action_pos_emb = nn.Parameter(torch.randn(1, max_len, self.fusion_dim))

        # ================= Temporal Transformer =================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # ================= Cross Attention =================
        self.window = 10
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.attn_ln = nn.LayerNorm(self.fusion_dim)

        # ================= Reward Head =================
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    # 计算一个segment的step reward
    def forward(self, system_feats, rgb_feats, depth_feats):
        
        # ================= Trajectory =================
        if system_feats.ndim == 2:
            system_feats = system_feats.unsqueeze(0)
            rgb_feats = rgb_feats.unsqueeze(0)
            depth_feats = depth_feats.unsqueeze(0)

        # system_feats: (B, T, 16)
        # rgb_feats:    (B, T, 384, 16, 16)
        # depth_feats:  (B, T, 3, 36, 64)

        B, T, _ = system_feats.shape

        #  Split State / Action 
        state = system_feats[..., :-2]   # (B,T,14)
        action = system_feats[..., -2:]  # (B,T,2)

        # ================= Encode State =================

        # sys
        sys = self.system_encoder(state.view(B*T, -1)).view(B, T, -1) # (B，T, system_embed_dim))

        # RGB 
        rgb = self.rgb_encoder(rgb_feats.view(B*T, 384, 16, 16))  # (B*T, rgb_embed_dim, 4, 4)
        rgb = rgb.flatten(2).transpose(1, 2)  # (B*T, 16, rgb_embed_dim)

        # Depth
        depth = self.depth_encoder(depth_feats.view(B*T, 3, 36, 64))  # (B*T, depth_embed_dim, 4, 4)
        depth = depth.flatten(2).transpose(1, 2)  # (B*T, 16, depth_embed_dim)

        # Vision Tokens
        vision = torch.cat([rgb, depth], dim=1)  # (B*T, 32, vision_dim) vision_dim = depth_embed_dim = rgb_embed_dim

        # Project to Unified Dim
        vision_embed = self.vision_proj(vision)  # (B*T, 32, F)
        sys_embed = self.state_proj(sys)  # (B, T, F)

        # Reshape to Temporal
        vision_embed = vision_embed.view(B, T, 32, -1)   # (B,T,32,F)
        sys_embed = sys_embed.unsqueeze(2) # (B, T, 1, F)

        # Positional Encoding
        sys_embed = sys_embed + self.state_pos_emb[:, :T, :1, :] # (B, T, 1, F)
        vision_embed = vision_embed + self.vision_pos_emb[:, :T, :32, :] # (B, T, 32, F)

        #  Merge Tokens 
        fused_state = torch.cat([sys_embed, vision_embed], dim=2)  # (B,T,33,F)

        # ================= Temporal-Spatial Transformer =================
        B, T, N, F = fused_state.shape   # N = 33

        # flatten temporal + spatial tokens
        fused_state = fused_state.reshape(B, T * N, F)   # (B, T*N, F)

        # Window causal mask
        token_idx = torch.arange(T * N, device=fused_state.device)

        # token -> timestep
        time_idx = token_idx // N

        i = time_idx.unsqueeze(1)
        j = time_idx.unsqueeze(0)

        mask = (j <= i) & (j >= i - self.window)
        mask = torch.where(mask, 0.0, float('-inf'))

        # Temporal-Spatial Self Attention
        fused_state = self.temporal_transformer(
            fused_state,
            mask=mask
        )   # (B, T*N, F)

        # Pool spatial tokens
        fused_state = fused_state.view(B, T, N, F) # (B,T,N,F)

        fused_state = fused_state.mean(dim=2) # (B,T,F)
        
        # ================= Action-conditioned attention =================

        # Encode Action
        action_embed = self.action_encoder(action.view(B*T, -1)).view(B, T, -1)  # (B,T,A)
        action_embed = self.action_proj(action_embed)                             # (B,T,F)
        action_embed = action_embed + self.action_pos_emb[:, :T, :]              # (B,T,F)

        # Window Causal Mask
        i = torch.arange(T, device=fused_state.device).unsqueeze(1)  # (T,1)
        j = torch.arange(T, device=fused_state.device).unsqueeze(0)  # (1,T)

        mask = (j <= i) & (j >= i - self.window)  # (T,T)
        mask = torch.where(mask, 0.0, float('-inf'))

        # Cross Attention
        attn_out, _ = self.cross_attn(
            query=action_embed,   # (B,T,F)
            key=fused_state,      # (B,T,F)
            value=fused_state,    # (B,T,F)
            attn_mask=mask
        )  # (B,T,F)

        # Residual
        temporal_feat = self.attn_ln(attn_out + action_embed)  # (B,T,F)

        # Reward Head
        step_rewards = self.head(temporal_feat).squeeze(-1)  # (B,T)

        return step_rewards
    
class RewardNet_Traj_Multimodal_SelfAttn(nn.Module):

    """
    输入：
        system_feats: (B, T, D)
        rgb_feats:    (B, T, 384, 16, 16)
        depth_feats:  (B, T, 3, 36, 64)
    输出：
        segment reward: (B, scalar)
    """
    def __init__(
        self,
        system_dim =16,
        rgb_embed_dim=256,
        depth_embed_dim=64,
        system_embed_dim=64,
        max_len=200,
    ):
        super().__init__()

        # ================= System Encoder =================
        self.system_encoder = nn.Sequential(
            nn.Linear(system_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, system_embed_dim),
            nn.LayerNorm(system_embed_dim),
            nn.ReLU(inplace=True)
        )

        # ================= RGB Encoder =================
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), # 变为 (B*T, 128, 1, 1)
            nn.Flatten(),
            nn.Linear(128, rgb_embed_dim),
            nn.LayerNorm(rgb_embed_dim),
            nn.ReLU(inplace=True)
        )

        # ================= Depth Encoder =================
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, depth_embed_dim),
            nn.ReLU(inplace=True)
        )

        # ================= Fusion =================
        self.fusion_dim = system_embed_dim + rgb_embed_dim + depth_embed_dim

        # ================= Positional Encoding =================
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, self.fusion_dim))

        # ================= Transformer =================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.fusion_dim,
            nhead=4,
            batch_first=True,
            dim_feedforward=512,
            dropout=0.1
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ================= Reward Head =================
        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    # 计算一个segment的sum reward
    def forward(self, system_feats, rgb_feats, depth_feats):
        
        if system_feats.ndim == 2:
            system_feats = system_feats.unsqueeze(0)
            rgb_feats = rgb_feats.unsqueeze(0)
            depth_feats = depth_feats.unsqueeze(0)

        B, T, _ = system_feats.shape

        sys_embed = self.system_encoder(system_feats.view(B*T, -1)).view(B, T, -1)
        rgb_embed = self.rgb_encoder(rgb_feats.view(B*T, 384, 16, 16)).view(B, T, -1)
        depth_embed = self.depth_encoder(depth_feats.view(B*T, 3, 36, 64)).view(B, T, -1)

        fused = torch.cat([sys_embed, rgb_embed, depth_embed], dim=-1) # (B,T,384)
        fused = fused + self.pos_emb[:, :T, :] # (B,T,384)

        mask = torch.triu(torch.ones(T, T, device=fused.device) * float('-inf'), diagonal=1) # (T,T)
        temporal_feat = self.temporal(fused, mask=mask) 
        step_rewards = self.head(temporal_feat).squeeze(-1) # (B,T,384) - (B,T,128) - (B,T,1)
        
        return step_rewards # (B, T)
    
class RewardNet_Traj_Sys_SelfAttn(nn.Module):
    """CrowdSim reward net: vec-obs + local-map CNN + depth CNN + action → reward.

    Architecture (inspired by RewardNet_Traj_Multimodal_CrossAttn):

    State path  — encodes each modality independently then fuses:
        vec_encoder  : Linear(30 → 64)
        map_encoder  : CNN(1,24,24 → 64)
        depth_encoder: CNN(1,H,W → 64)   [AdaptiveAvgPool → size-agnostic]
        state_fusion : Linear(192 → 128) → Temporal Self-Attn(causal)

    Action path — action cross-attends against state features:
        action_encoder: Linear(2 → 128)
        cross_attn    : MultiheadAttn(query=action, key/value=state, causal)
        → residual + LayerNorm

    Reward head : Linear(128→64) → Linear(64→1) → Tanh  →  (B, T)

    Parameters
    ----------
    vec_dim        : dimension of the vector-obs slice (default 30)
    map_size       : local-map side length in pixels (default 24)
    action_dim     : action dimension (default 2)
    depth_embed_dim: depth CNN output channels (default 64)
    state_embed_dim: state-path embedding width (default 128)
    action_embed_dim: action-path embedding width (default 128)
    max_len        : max sequence length for positional embeddings
    window         : not used (kept for API compatibility)
    """

    def __init__(
        self,
        vec_dim: int = 30,  # default matches rl_num_neighbors=4: 5+5+5*4=30
        map_size: int = 0,
        action_dim: int = 2,
        depth_embed_dim: int = 64,
        state_embed_dim: int = 128,
        action_embed_dim: int = 128,
        max_len: int = 200,
        **__,                    # absorb legacy / unknown kwargs (e.g. system_dim)
    ):
        super().__init__()
        self.vec_dim      = vec_dim
        self.map_size     = int(map_size)
        self.has_map      = self.map_size > 0
        self.map_flat_dim = self.map_size * self.map_size
        self.action_dim   = action_dim
        self.state_embed_dim  = state_embed_dim
        self.action_embed_dim = action_embed_dim

        # ── Vector encoder ──────────────────────────────────────────
        self.vec_encoder = nn.Sequential(
            nn.Linear(vec_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )  # (B*T, 64)

        # ── Local-map CNN (only when the obs carries an occupancy map) ──
        if self.has_map:
            self.map_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),            # 32×3×3 = 288 for map_size=24
                nn.Linear(288, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
            )  # (B*T, 64)
        else:
            self.map_encoder = None

        # ── Depth CNN ────────────────────────────────────────────────
        # AdaptiveAvgPool makes it agnostic to input resolution (32×32, 64×64, …)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, depth_embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),            # (B*T, depth_embed_dim)
            nn.Linear(depth_embed_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )  # (B*T, 64)

        # ── State fusion: cat(vec64, [map64,] depth64) → state_embed ──
        fusion_in = 64 + (64 if self.has_map else 0) + 64
        self.state_fusion = nn.Sequential(
            nn.Linear(fusion_in, state_embed_dim),
            nn.LayerNorm(state_embed_dim),
            nn.ReLU(inplace=True),
        )  # (B*T, state_embed_dim)

        # ── Action encoder ───────────────────────────────────────────
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.ReLU(inplace=True),
        )  # (B*T, action_embed_dim)

        # ── Positional embeddings ────────────────────────────────────
        self.state_pos_emb  = nn.Parameter(torch.randn(1, max_len, state_embed_dim))
        self.action_pos_emb = nn.Parameter(torch.randn(1, max_len, action_embed_dim))

        # ── State temporal self-attention ────────────────────────────
        state_layer = nn.TransformerEncoderLayer(
            d_model=state_embed_dim,
            nhead=4,
            batch_first=True,
            dim_feedforward=512,
            dropout=0.1,
        )
        self.state_temporal = nn.TransformerEncoder(state_layer, num_layers=2)

        # ── Action cross-attention against state features ────────────
        # If dims differ, project state to action_embed_dim first
        self.state_proj = (
            nn.Linear(state_embed_dim, action_embed_dim)
            if state_embed_dim != action_embed_dim else nn.Identity()
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=action_embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_attn_ln = nn.LayerNorm(action_embed_dim)

        # ── Reward head ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(action_embed_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        system_feats: torch.Tensor,              # (B, T, F)  [vec(|map)|action]
        depth_feats: "torch.Tensor | None" = None,  # (B, T, H, W) or None
        rgb_feats=None,                          # unused, kept for API compat
    ) -> torch.Tensor:                           # (B, T)
        if system_feats.ndim == 2:
            system_feats = system_feats.unsqueeze(0)
        B, T, _ = system_feats.shape

        # ── Split system_feats using absolute indices ─────────────────
        # Layout: [vec(vec_dim) | map(map_flat_dim) | action(action_dim)]
        # When has_map is False, map_flat_dim == 0 and act_start == vec_dim.
        act_start = self.vec_dim + self.map_flat_dim
        vec      = system_feats[..., :self.vec_dim]              # (B, T, vec_dim)
        act      = system_feats[..., act_start:]                 # (B, T, action_dim)

        # ── Encode vec (+ map when present) ───────────────────────────
        v = self.vec_encoder(vec.reshape(B * T, -1))            # (B*T, 64)
        if self.has_map:
            map_flat = system_feats[..., self.vec_dim:act_start]  # (B, T, map_flat_dim)
            m = self.map_encoder(
                map_flat.reshape(B * T, 1, self.map_size, self.map_size)
            )                                                      # (B*T, 64)
            vec_map = torch.cat([v, m], dim=-1)
        else:
            vec_map = v

        # ── Encode depth ──────────────────────────────────────────────
        if depth_feats is not None:
            if depth_feats.ndim == 4:                         # (B, T, H, W)
                H, W = depth_feats.shape[-2], depth_feats.shape[-1]
                d_in = depth_feats.reshape(B * T, 1, H, W)
            else:                                             # (B*T, H, W)
                d_in = depth_feats.unsqueeze(1)
            d = self.depth_encoder(d_in)                      # (B*T, 64)
        else:
            d = torch.zeros(B * T, 64, device=system_feats.device)

        # ── State fusion → temporal self-attention ────────────────────
        state = self.state_fusion(
            torch.cat([vec_map, d], dim=-1)
        ).view(B, T, -1)                                      # (B, T, state_embed)
        state = state + self.state_pos_emb[:, :T, :]

        causal = self._causal_mask(T, system_feats.device)
        state_out = self.state_temporal(state, mask=causal)   # (B, T, state_embed)
        state_out = self.state_proj(state_out)                # (B, T, action_embed)

        # ── Action cross-attention ────────────────────────────────────
        action = self.action_encoder(act.reshape(B * T, -1)).view(B, T, -1)
        action = action + self.action_pos_emb[:, :T, :]       # (B, T, action_embed)

        attn_out, _ = self.cross_attn(
            query=action,
            key=state_out,
            value=state_out,
            attn_mask=causal,
        )
        feat = self.cross_attn_ln(attn_out + action)          # (B, T, action_embed)

        return self.head(feat).squeeze(-1)                    # (B, T)