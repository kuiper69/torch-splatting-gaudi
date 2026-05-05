import torch
import torch.nn as nn
import math
import habana_frameworks.torch.core as htcore

from einops import reduce


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance


def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the screen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    W = viewmatrix[:3,:3].T  # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)

    # add low pass filter here according to E.q. 32
    filter = torch.eye(2, 2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]


def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points)  # object space
    points_h = points_o @ viewmatrix @ projmatrix  # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()


@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from .utils.sh_utils import eval_sh
import torch.autograd.profiler as profiler
USE_PROFILE = False
import contextlib


class GaussRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, active_sh_degree=3, white_bkgd=True,
                 image_height=256, image_width=256, **kwargs):
        super(GaussRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.image_height = image_height
        self.image_width = image_width
        self.TILE_SIZE = 64
        self.P_MAX = 2048

        assert image_height % self.TILE_SIZE == 0, \
            f"image_height {image_height} must be divisible by TILE_SIZE {self.TILE_SIZE}"
        assert image_width % self.TILE_SIZE == 0, \
            f"image_width {image_width} must be divisible by TILE_SIZE {self.TILE_SIZE}"

        tiles_h = image_height // self.TILE_SIZE
        tiles_w = image_width // self.TILE_SIZE
        self.tiles_h = tiles_h
        self.tiles_w = tiles_w
        self.num_tiles = tiles_h * tiles_w

        # Tile boundary tensors — precomputed on CPU, moved to device on first render
        tile_h_idx, tile_w_idx = torch.meshgrid(
            torch.arange(tiles_h), torch.arange(tiles_w), indexing='ij'
        )
        h_mins = (tile_h_idx * self.TILE_SIZE).flatten().float()  # [num_tiles]
        w_mins = (tile_w_idx * self.TILE_SIZE).flatten().float()  # [num_tiles]
        self._tile_h_mins = h_mins
        self._tile_w_mins = w_mins
        self._tile_h_maxs = h_mins + self.TILE_SIZE - 1
        self._tile_w_maxs = w_mins + self.TILE_SIZE - 1

        # Pixel coordinates per tile — [num_tiles, TILE_SIZE², 2]
        local_x = torch.arange(self.TILE_SIZE).float()
        local_y = torch.arange(self.TILE_SIZE).float()
        local_xx, local_yy = torch.meshgrid(local_x, local_y, indexing='xy')
        local_flat = torch.stack([local_xx, local_yy], dim=-1).flatten(0, 1)  # [TILE_SIZE², 2]
        offsets = torch.stack([w_mins, h_mins], dim=-1)                        # [num_tiles, 2]
        self._tile_coords = local_flat.unsqueeze(0) + offsets.unsqueeze(1)     # [num_tiles, TILE_SIZE², 2]

        # Device-resident copies, populated on first call to _ensure_on_device
        self._precomputed_device = None
        self.tile_h_mins = None
        self.tile_w_mins = None
        self.tile_h_maxs = None
        self.tile_w_maxs = None
        self.tile_coords = None

    def _ensure_on_device(self, device):
        if self._precomputed_device != device:
            self.tile_h_mins = self._tile_h_mins.to(device)
            self.tile_w_mins = self._tile_w_mins.to(device)
            self.tile_h_maxs = self._tile_h_maxs.to(device)
            self.tile_w_maxs = self._tile_w_maxs.to(device)
            self.tile_coords = self._tile_coords.to(device)
            self._precomputed_device = device

    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color

    def render_tensors(self, means2D, cov2d, color, opacity, depths):
        """
        Pure tensor interface for the batched tile rasterization pipeline.
        Suitable for make_graphed_callables capture.

        Args:
            means2D:  [N, 2]    float32 — projected 2D Gaussian centers
            cov2d:    [N, 2, 2] float32 — 2D covariance matrices
            color:    [N, 3]    float32 — per-Gaussian RGB color
            opacity:  [N, 1]    float32 — per-Gaussian opacity
            depths:   [N]       float32 — per-Gaussian view-space depth

        Returns:
            render_color: [H, W, 3]
            render_depth: [H, W, 1]
            render_alpha: [H, W, 1]
        """
        device = means2D.device
        self._ensure_on_device(device)

        # Stage 1: overlap mask [num_tiles, N]
        rect_min, rect_max = get_rect(
            means2D, get_radius(cov2d),
            width=self.image_width,
            height=self.image_height
        )
        over_tl_x = torch.maximum(rect_min[None, :, 0], self.tile_w_mins[:, None])
        over_tl_y = torch.maximum(rect_min[None, :, 1], self.tile_h_mins[:, None])
        over_br_x = torch.minimum(rect_max[None, :, 0], self.tile_w_maxs[:, None])
        over_br_y = torch.minimum(rect_max[None, :, 1], self.tile_h_maxs[:, None])
        overlap_mask = (over_br_x > over_tl_x) & (over_br_y > over_tl_y)  # [num_tiles, N]

        # Stage 2: sort with sentinel depth, truncate to P_MAX
        depths_tiled = depths.unsqueeze(0).repeat(self.num_tiles, 1)
        depths_tiled = torch.where(overlap_mask, depths_tiled,
                                   torch.full_like(depths_tiled, 1e10))
        sorted_depths_full, sort_indices = torch.sort(depths_tiled, dim=1)  # [num_tiles, N]
        sort_indices  = sort_indices[:, :self.P_MAX]         # [num_tiles, P_MAX]
        sorted_depths = sorted_depths_full[:, :self.P_MAX]   # [num_tiles, P_MAX]
        valid_mask    = sorted_depths < 1e9                  # [num_tiles, P_MAX]

        # Stage 3: gather per-Gaussian attributes → [num_tiles, P_MAX, D]
        def gather_attr(x, trailing):
            idx = sort_indices
            for _ in trailing:
                idx = idx.unsqueeze(-1)
            idx = idx.expand(-1, -1, *trailing)
            x_exp = x.unsqueeze(0).repeat(self.num_tiles, *([1] * len(x.shape))).contiguous()
            return torch.gather(x_exp, 1, idx)

        sorted_means2D = gather_attr(means2D, (2,))    # [num_tiles, P_MAX, 2]
        sorted_cov2d   = gather_attr(cov2d,   (2, 2))  # [num_tiles, P_MAX, 2, 2]
        sorted_opacity = gather_attr(opacity, (1,))    # [num_tiles, P_MAX, 1]
        sorted_color   = gather_attr(color,   (3,))    # [num_tiles, P_MAX, 3]

        # Stage 4: closed-form 2x2 inverse (TPC-friendly, replaces .inverse())
        a = sorted_cov2d[..., 0, 0]
        b = sorted_cov2d[..., 0, 1]
        c = sorted_cov2d[..., 1, 0]
        d = sorted_cov2d[..., 1, 1]
        det = (a * d - b * c).clamp(min=1e-6)
        inv_det = 1.0 / det
        sorted_conic = torch.stack(
            [d * inv_det, -b * inv_det, -c * inv_det, a * inv_det], dim=-1
        ).reshape(*sorted_cov2d.shape)  # [num_tiles, P_MAX, 2, 2]

        # dx: [num_tiles, TILE_SIZE², P_MAX, 2]
        dx = self.tile_coords.unsqueeze(2) - sorted_means2D.unsqueeze(1)

        # gauss_weight: [num_tiles, TILE_SIZE², P_MAX]
        gauss_weight = torch.exp(-0.5 * (
            dx[..., 0]**2 * sorted_conic[:, None, :, 0, 0]
            + dx[..., 1]**2 * sorted_conic[:, None, :, 1, 1]
            + dx[..., 0] * dx[..., 1] * sorted_conic[:, None, :, 0, 1]
            + dx[..., 0] * dx[..., 1] * sorted_conic[:, None, :, 1, 0]
        ))

        # Stage 5: alpha compositing
        valid = valid_mask[:, None, :].float()  # [num_tiles, 1, P_MAX]
        alpha = (gauss_weight * sorted_opacity[:, None, :, 0]).clamp(max=0.99) * valid
        T = torch.cat(
            [torch.ones_like(alpha[:, :, :1]), 1 - alpha[:, :, :-1]], dim=2
        ).cumprod(dim=2)
        acc_alpha = (alpha * T).sum(dim=2)  # [num_tiles, TILE_SIZE²]

        tile_color = ((T * alpha).unsqueeze(-1) * sorted_color[:, None, :, :]).sum(dim=2)
        tile_color = tile_color + (1 - acc_alpha.unsqueeze(-1)) * (1 if self.white_bkgd else 0)
        tile_depth = ((T * alpha) * sorted_depths[:, None, :]).sum(dim=2)

        # Reconstruct full image [H, W, C]
        render_color = (
            tile_color
            .reshape(self.tiles_h, self.tiles_w, self.TILE_SIZE, self.TILE_SIZE, 3)
            .permute(0, 2, 1, 3, 4)
            .reshape(self.image_height, self.image_width, 3)
        )
        render_depth = (
            tile_depth
            .reshape(self.tiles_h, self.tiles_w, self.TILE_SIZE, self.TILE_SIZE)
            .permute(0, 2, 1, 3)
            .reshape(self.image_height, self.image_width, 1)
        )
        render_alpha = (
            acc_alpha
            .reshape(self.tiles_h, self.tiles_w, self.TILE_SIZE, self.TILE_SIZE)
            .permute(0, 2, 1, 3)
            .reshape(self.image_height, self.image_width, 1)
        )

        return render_color, render_depth, render_alpha

    def render(self, camera, means2D, cov2d, color, opacity, depths):
        radii = get_radius(cov2d)

        render_color, render_depth, render_alpha = self.render_tensors(
            means2D, cov2d, color, opacity, depths
        )

        return {
            "render": render_color,
            "depth": render_depth,
            "alpha": render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }

    def forward(self, camera, pc, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features

        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext

        with prof("projection"):
            mean_ndc, mean_view, in_mask = projection_ndc(
                means3D,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.projection_matrix
            )
            mean_ndc = mean_ndc * in_mask.unsqueeze(-1)
            mean_view = mean_view * in_mask.unsqueeze(-1)
            depths = mean_view[:, 2]

        with prof("build color"):
            color = self.build_color(means3D=means3D, shs=shs, camera=camera)

        with prof("build cov3d"):
            cov3d = build_covariance_3d(scales, rotations)

        with prof("build cov2d"):
            cov2d = build_covariance_2d(
                mean3d=means3D,
                cov3d=cov3d,
                viewmatrix=camera.world_view_transform,
                fov_x=camera.FoVx,
                fov_y=camera.FoVy,
                focal_x=camera.focal_x,
                focal_y=camera.focal_y
            )
            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        with prof("render"):
            rets = self.render(
                camera=camera,
                means2D=means2D,
                cov2d=cov2d,
                color=color,
                opacity=opacity,
                depths=depths,
            )

        return rets
