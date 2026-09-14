"""Visual-only SMPL mesh overlay for ProtoMotions IsaacLab humanoids."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from protomotions.utils.rotations import (
    exp_map_to_quat,
    matrix_to_quaternion,
    quat_to_exp_map,
    quaternion_to_matrix,
)


SMPL_JOINT_NAMES: list[str] = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

MJCF_JOINT_NAMES: list[str] = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]


def _parse_optional_int_env(name: str) -> int | None:
    """Read an optional non-negative int from an environment variable."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        v = int(raw)
        return v if v >= 0 else None
    except ValueError:
        return None


@dataclass
class HumanMeshConfig:
    model_dir: str
    num_envs: int
    device: str
    prim_root: str = "/World/CrowdSim/HumanMesh"
    color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    opacity: float = 1.0
    hide_humanoid: bool = False
    texture_path: str | None = None
    texture_paths: tuple[str, ...] = ()
    appearance_seed: int = 42
    shape_std: float = 0.55
    shape_clip: float = 1.25
    # Number of active (navigating) humanoids.  Only these are hidden when
    # hide_humanoid is true; frozen filler slots (env_id >= num_active_humanoids)
    # stay visible so the depth camera can still see them as obstacles.
    num_active_humanoids: int | None = None


class ProtoMotionsHumanMeshAdapter:
    """Adapter between ProtoMotions IsaacLab state and the SMPL mesh overlay."""

    def __init__(self, simulator, visualizer: "SMPLMeshVisualizer") -> None:
        self.simulator = simulator
        self.visualizer = visualizer
        self.pose_mapper = SMPLRobotPoseMapper()

    @classmethod
    def from_simulator(cls, simulator) -> "ProtoMotionsHumanMeshAdapter":
        from CrowdSim.scene_setup import resolve_repo_path
        model_dir = os.environ.get(
            "CROWDSIM_SMPL_MODEL_DIR", str(resolve_repo_path("data/smpl"))
        )
        texture_path = os.environ.get("CROWDSIM_SMPL_TEXTURE_PATH")
        if not texture_path:
            for name in ("smpl_body_texture.png", "smpl_uv_20200910.png"):
                candidate = Path(model_dir) / name
                if candidate.exists():
                    texture_path = str(candidate)
                    break
        texture_dir = Path(
            os.environ.get("CROWDSIM_SMPL_TEXTURE_DIR", model_dir)
        ).expanduser()
        texture_paths: list[str] = []
        if texture_path:
            texture_paths.append(str(Path(texture_path).expanduser()))
        if texture_dir.is_dir():
            supported_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            texture_paths.extend(
                str(path)
                for path in sorted(texture_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in supported_suffixes
            )
        texture_paths = list(dict.fromkeys(texture_paths))
        cfg = HumanMeshConfig(
            model_dir=model_dir,
            num_envs=int(getattr(simulator, "num_envs", 1)),
            device=str(getattr(simulator, "device", "cuda:0")),
            hide_humanoid=os.environ.get("CROWDSIM_HIDE_HUMANOID", "0") == "1",
            texture_path=texture_path,
            texture_paths=tuple(texture_paths),
            appearance_seed=int(os.environ.get("CROWDSIM_HUMAN_APPEARANCE_SEED", "42")),
            shape_std=float(os.environ.get("CROWDSIM_SMPL_SHAPE_STD", "0.55")),
            shape_clip=float(os.environ.get("CROWDSIM_SMPL_SHAPE_CLIP", "1.25")),
            num_active_humanoids=_parse_optional_int_env("CROWDSIM_NUM_ACTIVE_HUMANOIDS"),
        )
        return cls(simulator=simulator, visualizer=SMPLMeshVisualizer.from_config(cfg))

    def create(self) -> None:
        self.visualizer.create()
        if self.visualizer.cfg.hide_humanoid:
            self._hide_humanoid_visuals()

    def update(self) -> None:
        robot = self.simulator._robot
        body_pose = self.pose_mapper(robot.data.joint_pos, list(robot.data.joint_names))
        self.visualizer.update(
            body_pose=body_pose,
            root_pos=robot.data.root_pos_w,
            root_quat_wxyz=robot.data.root_quat_w,
        )

    def _hide_humanoid_visuals(self) -> None:
        from pxr import UsdGeom

        stage = self.visualizer.stage
        if stage is None:
            return

        # Only hide ACTIVE humanoid slots.  Frozen filler slots (env_id >=
        # num_active_humanoids) must stay visible so the active robot's depth
        # camera can still perceive them as obstacles — otherwise the policy
        # has no observation signal for a physical body it can collide with.
        n_active = self.visualizer.cfg.num_active_humanoids
        if n_active is None:
            n_active = self.visualizer.cfg.num_envs
        for env_id in range(min(n_active, self.visualizer.cfg.num_envs)):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()


class SMPLRobotPoseMapper:
    """Convert ProtoMotions SMPL robot exp-map DOFs into SMPL body_pose."""

    def __init__(self) -> None:
        self.smpl_to_robot = smpl_to_robot_matrix()
        self.body_names = [name for name in SMPL_JOINT_NAMES if name != "Pelvis"]

    def __call__(self, joint_pos: torch.Tensor, joint_names: Sequence[str]) -> torch.Tensor:
        if joint_pos.ndim != 2:
            raise ValueError(f"joint_pos must be [N, D], got {tuple(joint_pos.shape)}")

        by_name = self._map_by_name(joint_pos, joint_names)
        if by_name is not None:
            return by_name
        return self._map_by_mjcf_order(joint_pos)

    def _map_by_name(
        self, joint_pos: torch.Tensor, joint_names: Sequence[str]
    ) -> Optional[torch.Tensor]:
        rotvecs: list[torch.Tensor] = []
        for body_name in self.body_names:
            indices = self._body_triplet_indices(body_name, joint_names)
            if indices is None:
                return None
            robot_expmap = joint_pos[:, indices]
            rotvecs.append(robot_expmap_to_smpl_rotvec(robot_expmap, self.smpl_to_robot))
        return torch.cat(rotvecs, dim=-1).contiguous()

    def _map_by_mjcf_order(self, joint_pos: torch.Tensor) -> torch.Tensor:
        expected_dofs = 3 * (len(MJCF_JOINT_NAMES) - 1)
        if joint_pos.shape[1] < expected_dofs:
            raise RuntimeError(
                f"Expected at least {expected_dofs} SMPL robot DOFs, got {joint_pos.shape[1]}"
            )

        body_to_rotvec: dict[str, torch.Tensor] = {}
        cursor = 0
        for body_name in MJCF_JOINT_NAMES:
            if body_name == "Pelvis":
                continue
            robot_expmap = joint_pos[:, cursor : cursor + 3]
            body_to_rotvec[body_name] = robot_expmap_to_smpl_rotvec(
                robot_expmap, self.smpl_to_robot
            )
            cursor += 3

        return torch.cat([body_to_rotvec[name] for name in self.body_names], dim=-1)

    @staticmethod
    def _body_triplet_indices(
        body_name: str, joint_names: Sequence[str]
    ) -> Optional[list[int]]:
        axis_to_index: dict[str, int] = {}
        prefix = f"{body_name}_"
        for index, joint_name in enumerate(joint_names):
            if joint_name.startswith(prefix):
                axis_to_index[joint_name[len(prefix) :].lower()] = index

        if set(axis_to_index) != {"x", "y", "z"}:
            return None
        return [axis_to_index["x"], axis_to_index["y"], axis_to_index["z"]]


class SMPLMeshVisualizer:
    """Create and update batched UsdGeom.Mesh prims from SMPL vertices."""

    def __init__(
        self,
        cfg: HumanMeshConfig,
        smpl_model,
        faces: np.ndarray,
        uvs: np.ndarray | None = None,
        uv_faces: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.smpl_model = smpl_model.to(self.device).eval()
        self.faces = np.asarray(faces, dtype=np.int64)
        self.uvs = np.asarray(uvs, dtype=np.float32) if uvs is not None else None
        self.uv_faces = np.asarray(uv_faces, dtype=np.int64) if uv_faces is not None else None
        self.texture_paths = tuple(
            path
            for path in (
                cfg.texture_paths
                or ((cfg.texture_path,) if cfg.texture_path else ())
            )
            if Path(path).is_file()
        )
        self.has_texture = self.uvs is not None and bool(self.texture_paths)
        rng = np.random.default_rng(int(cfg.appearance_seed))
        if self.texture_paths:
            # Shuffle without replacement within each cycle, then reshuffle if
            # there are more people than textures. This is deterministic for a
            # given appearance_seed while avoiding sorted/modulo assignment.
            assigned: list[str] = []
            while len(assigned) < cfg.num_envs:
                order = rng.permutation(len(self.texture_paths))
                assigned.extend(self.texture_paths[int(index)] for index in order)
            self.person_texture_paths = tuple(assigned[:cfg.num_envs])
        else:
            self.person_texture_paths = ()
        shape_scale = np.asarray(
            [1.0, 0.85, 0.85, 0.7, 0.7, 0.55, 0.55, 0.45, 0.45, 0.4],
            dtype=np.float32,
        )
        betas = rng.normal(
            0.0, float(cfg.shape_std), size=(cfg.num_envs, 10)
        ).astype(np.float32)
        betas *= shape_scale[None]
        self.betas = torch.as_tensor(
            np.clip(betas, -float(cfg.shape_clip), float(cfg.shape_clip)),
            device=self.device,
        )
        self.shape_floor_offsets = self._compute_shape_floor_offsets()
        self.stage = None
        self._points_attrs: list = []
        self._created = False

    @torch.no_grad()
    def _compute_shape_floor_offsets(self) -> torch.Tensor:
        """Match each shaped mesh's rest-pose floor to neutral SMPL."""
        dtype = self.betas.dtype
        zeros_orient = torch.zeros(
            (self.cfg.num_envs, 3), device=self.device, dtype=dtype
        )
        zeros_pose = torch.zeros(
            (self.cfg.num_envs, 3 * (len(SMPL_JOINT_NAMES) - 1)),
            device=self.device,
            dtype=dtype,
        )
        common = {
            "global_orient": zeros_orient,
            "body_pose": zeros_pose,
            "transl": zeros_orient,
            "return_verts": True,
        }
        shaped = self.smpl_model(betas=self.betas, **common)
        neutral = self.smpl_model(
            betas=torch.zeros_like(self.betas), **common
        )
        transform = smpl_to_robot_matrix(self.device, dtype).T

        def local_floor(output) -> torch.Tensor:
            centered = output.vertices - output.joints[:, 0:1, :]
            robot_vertices = torch.matmul(centered, transform)
            return robot_vertices[..., 2].amin(dim=1)

        return (local_floor(neutral) - local_floor(shaped)).contiguous()

    @classmethod
    def from_config(cls, cfg: HumanMeshConfig) -> "SMPLMeshVisualizer":
        install_smpl_pickle_compat()
        try:
            import smplx
        except ImportError as exc:
            raise ImportError("Install smplx and place SMPL_*.pkl files in data/smpl.") from exc

        smpl_model = smplx.create(
            resolve_smpl_model_path(cfg.model_dir),
            model_type="smpl",
            gender="neutral",
            num_betas=10,
            use_pca=False,
            batch_size=cfg.num_envs,
        )

        uvs: np.ndarray | None = None
        uv_faces: np.ndarray | None = None
        uv_npz = Path(cfg.model_dir) / "smpl_uv.npz"
        if uv_npz.exists() and (cfg.texture_path or cfg.texture_paths):
            try:
                data = np.load(str(uv_npz), allow_pickle=False)
                uvs = data["uvs"]
                uv_faces = data["uv_faces"]
            except Exception as exc:
                # smpl_uv.npz 可能是 Git LFS 指针（未 pull）或格式不正确；
                # 降级为无纹理模式，不崩溃程序。
                # 运行 `git lfs pull --include="data/smpl/smpl_uv.npz"` 获取真实数据，
                # 或运行 tools/generate_smpl_uv.py 重新生成。
                print(
                    f"[SMPLMesh] WARNING: failed to load UV data from {uv_npz}: {exc}\n"
                    "  Falling back to solid-color mesh (no texture). "
                    "Run `git lfs pull` or `tools/generate_smpl_uv.py` to fix."
                )

        return cls(
            cfg,
            smpl_model,
            np.asarray(smpl_model.faces, dtype=np.int64),
            uvs=uvs,
            uv_faces=uv_faces,
        )

    def _create_person_material(self, env_id: int, Gf, Sdf, UsdShade):
        material_root = f"{self.cfg.prim_root}/Materials/person_{env_id}"
        material = UsdShade.Material.Define(self.stage, material_root)
        shader = UsdShade.Shader.Define(self.stage, f"{material_root}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
            float(self.cfg.opacity)
        )
        if self.has_texture:
            tex_path = Path(self.person_texture_paths[env_id]).resolve()
            st_reader = UsdShade.Shader.Define(
                self.stage, f"{material_root}/StReader"
            )
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            tex_shader = UsdShade.Shader.Define(
                self.stage, f"{material_root}/TexShader"
            )
            tex_shader.CreateIdAttr("UsdUVTexture")
            tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex_path))
            tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                st_reader.ConnectableAPI(), "result"
            )
            tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
            st_input = material.CreateInput(
                "frame:stPrimvarName", Sdf.ValueTypeNames.Token
            )
            st_input.Set("st")
            st_reader.CreateInput(
                "varname", Sdf.ValueTypeNames.Token
            ).ConnectToSource(st_input)
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*self.cfg.color)
            )

        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        return material

    def create(self) -> None:
        if self._created:
            return

        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("Could not get active USD stage.")

        UsdGeom.Xform.Define(self.stage, self.cfg.prim_root)

        # ── Mesh geometry ─────────────────────────────────────────
        num_vertices = int(getattr(self.smpl_model, "v_template").shape[0])
        zero_points = np.zeros((num_vertices, 3), dtype=np.float32)
        face_vertex_counts = [3] * int(self.faces.shape[0])
        face_vertex_indices = self.faces.reshape(-1).astype(np.int64).tolist()

        for env_id in range(self.cfg.num_envs):
            material = self._create_person_material(env_id, Gf, Sdf, UsdShade)
            mesh_path = f"{self.cfg.prim_root}/env_{env_id}/Body"
            UsdGeom.Xform.Define(self.stage, f"{self.cfg.prim_root}/env_{env_id}")
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
            mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
            mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*self.cfg.color)]))
            mesh.CreateDisplayOpacityAttr([float(self.cfg.opacity)])
            self._points_attrs.append(mesh.CreatePointsAttr(to_vt_vec3f(zero_points)))

            # ── UV primvar (faceVarying — preserves seam discontinuities) ──
            #
            # 修复说明（Bug 2 + Bug 3）：
            #
            # Bug 2 — 错误的 V 轴翻转：
            #   原代码: 1.0 - vtx_uv[i, 1]
            #   USD UsdUVTexture 规范: st=(0,0) = 纹理左下角 (OpenGL 约定)，
            #   USD 在读取 PNG 时会内部翻转图像，使其符合 OpenGL 约定，
            #   因此应用方无需再手动翻转 V 轴。
            #   SMPL 官方 UV 数据同样使用 OpenGL 约定 (V=0 在底部)。
            #   双重翻转导致人脸区域 (V≈0.65-1.0) 被映射到纹理底部 (脚/手)。
            #
            # Bug 3 — vertex 插值在 UV 缝合线 (seam) 处的平均误差：
            #   SMPL 模型有多处 UV seam，seam 处同一几何顶点在不同面中
            #   对应完全不同的 UV 坐标（属于两个不相邻的 UV 岛）。
            #   简单平均后的 UV 落在两个岛中间，恰好可能位于人脸纹理区域，
            #   导致「人脸贴到肚子上」。
            #   修复：改用 faceVarying 插值，每个面的每个顶点角使用精确 UV，
            #   不做任何平均，完全保留 seam 处的不连续性。
            if self.has_texture and self.uvs is not None and self.uv_faces is not None:
                # faceVarying: 每个面×3个角有独立 UV，展开为 (N_faces*3,) 列表
                # uv_faces[f, c] → 在 uvs 中的 UV 顶点索引
                # uvs[uv_idx]    → (u, v)，SMPL OBJ/官方格式使用 OpenGL 约定
                #                  (V=0=底部)，与 USD 一致，无需翻转
                fv_uvs = self.uvs[self.uv_faces.reshape(-1)]  # (N_faces*3, 2)
                st_array = Vt.Vec2fArray([
                    Gf.Vec2f(float(fv_uvs[i, 0]), float(fv_uvs[i, 1]))
                    for i in range(fv_uvs.shape[0])
                ])
                primvars_api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
                st_primvar = primvars_api.CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.faceVarying,  # ← 关键修复：faceVarying 而非 vertex
                )
                st_primvar.Set(st_array)

            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

        self._created = True
        print(
            f"[SMPLMesh] Created {self.cfg.num_envs} distinct appearances "
            f"(texture_pool={len(self.texture_paths)}, random_assignment=true, "
            f"shape_std={self.cfg.shape_std:.2f}, "
            f"seed={self.cfg.appearance_seed}, "
            f"floor_offset=[{self.shape_floor_offsets.min().item():+.3f}, "
            f"{self.shape_floor_offsets.max().item():+.3f}]m)"
        )

    @torch.no_grad()
    def update(
        self,
        body_pose: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat_wxyz: torch.Tensor,
    ) -> None:
        if not self._created:
            self.create()

        body_pose = body_pose.to(self.device)
        root_pos = root_pos.to(self.device)
        root_quat_wxyz = root_quat_wxyz.to(self.device)
        zeros = torch.zeros((self.cfg.num_envs, 3), device=self.device, dtype=body_pose.dtype)
        betas = self.betas.to(dtype=body_pose.dtype)

        out = self.smpl_model(
            betas=betas,
            global_orient=zeros,
            body_pose=body_pose,
            transl=zeros,
            return_verts=True,
        )
        vertices = out.vertices - out.joints[:, 0:1, :]
        vertices = torch.matmul(vertices, smpl_to_robot_matrix(self.device, vertices.dtype).T)
        vertices[..., 2] += self.shape_floor_offsets.to(vertices.dtype)[:, None]
        vertices = quat_apply_wxyz(root_quat_wxyz[:, None, :], vertices) + root_pos[:, None, :]
        vertices = vertices.detach().cpu().numpy().astype(np.float32)

        for env_id, points_attr in enumerate(self._points_attrs):
            points_attr.Set(to_vt_vec3f(vertices[env_id]))


def smpl_to_robot_matrix(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed SMPL local frame -> ProtoMotions SMPL robot local frame transform."""
    rx90 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    yaw90 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    return yaw90 @ rx90


def robot_expmap_to_smpl_rotvec(
    robot_expmap: torch.Tensor,
    smpl_to_robot: torch.Tensor,
) -> torch.Tensor:
    robot_quat = exp_map_to_quat(robot_expmap, w_last=False)
    robot_rot = quaternion_to_matrix(robot_quat, w_last=False)
    transform = smpl_to_robot.to(device=robot_expmap.device, dtype=robot_expmap.dtype)
    smpl_rot = transform.T @ robot_rot @ transform
    smpl_quat = matrix_to_quaternion(smpl_rot, w_last=False)
    smpl_quat = smpl_quat / torch.clamp(smpl_quat.norm(dim=-1, keepdim=True), min=1e-8)
    return quat_to_exp_map(smpl_quat, w_last=False)


def quat_apply_wxyz(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz / torch.clamp(quat_wxyz.norm(dim=-1, keepdim=True), min=1e-8)
    q_vec = quat[..., 1:4]
    q_w = quat[..., 0:1]
    t = 2.0 * torch.cross(q_vec.expand_as(vec), vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec.expand_as(vec), t, dim=-1)


def resolve_smpl_model_path(model_dir: str) -> str:
    path = Path(model_dir).expanduser()
    if path.is_file():
        return str(path)
    smpl_sub = path / "smpl" / "SMPL_NEUTRAL.pkl"
    if smpl_sub.exists():
        return str(smpl_sub)
    flat_file = path / "SMPL_NEUTRAL.pkl"
    if flat_file.exists():
        return str(flat_file)
    return str(path)


def install_smpl_pickle_compat() -> None:
    """Patch Python 3.11 / NumPy aliases used by older SMPL pickle files."""
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    for name, value in {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def to_vt_vec3f(points: np.ndarray):
    from pxr import Gf, Vt

    points = np.asarray(points, dtype=np.float32)
    try:
        return Vt.Vec3fArray.FromNumpy(points)
    except Exception:
        return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in points])
