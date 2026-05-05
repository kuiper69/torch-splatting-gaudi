import torch
import habana_frameworks.torch.core as htcore
import numpy as np
import gaussian_splatting.utils as utils
from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer
from tqdm import tqdm
from pathlib import Path

USE_GPU_PYTORCH = True


class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        # Plain GaussRenderer — no make_graphed_callables wrapper
        # (render_tensors is captured inside the full HPUGraph below)
        self.gaussRender = GaussRenderer(**kwargs.get('render_kwargs', {}))
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0

    def on_train_step(self):
        # Used only indirectly via on_evaluate_step; not called in the
        # capture-replay training path.
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        rgb = self.data['rgb'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)
        out = self.gaussRender(pc=self.model, camera=camera)
        l1_loss   = loss_utils.l1_loss(out['render'], rgb)
        ssim_loss = 1.0 - loss_utils.ssim(out['render'], rgb)
        total_loss = (1 - self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss
        psnr = utils.img2psnr(out['render'], rgb)
        log_dict = {'total': total_loss, 'l1': l1_loss, 'ssim': ssim_loss, 'psnr': psnr}
        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)
        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd  = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth   = self.data['depth'][ind].detach().cpu().numpy()
        depth   = np.concatenate([depth, depth_pd], axis=1)
        depth   = (1 - depth / depth.max())
        depth   = plt.get_cmap('jet')(depth)[..., :3]
        image   = np.concatenate([rgb, rgb_pd], axis=1)
        image   = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)
        self.model.save_ply(self.results_folder / f'splats-{self.step}.ply')

    def train(self):
        device = 'hpu'
        H = self.gaussRender.image_height
        W = self.gaussRender.image_width

        # ------------------------------------------------------------------ #
        # Static tensors for per-iteration inputs that change each step.
        # Model parameters (nn.Parameters) are already at fixed HPU addresses
        # and get updated in-place by FusedAdamW — no static copies needed.
        # ------------------------------------------------------------------ #
        static_world_view     = torch.zeros(4, 4, device=device)
        static_proj_matrix    = torch.zeros(4, 4, device=device)
        static_camera_center  = torch.zeros(3,    device=device)
        static_rgb            = torch.zeros(H, W, 3, device=device)

        # Bake camera intrinsics from the first camera.
        # Assumption: FoVx, FoVy, focal_x, focal_y are shared across all
        # cameras in this dataset (same lens / same capture rig).
        # These values go into math.tan() calls in build_covariance_2d and
        # cannot be HPU tensors — they must be Python scalars baked at
        # capture time.
        _cam0 = to_viewpoint_camera(self.data['camera'][0])
        baked_fovx    = float(_cam0.FoVx)
        baked_fovy    = float(_cam0.FoVy)
        baked_focal_x = float(_cam0.focal_x)
        baked_focal_y = float(_cam0.focal_y)

        # ------------------------------------------------------------------ #
        # StaticCamera: a lightweight object whose extrinsic tensor attributes
        # point to the static HPU buffers above. Updated via copy_() before
        # each replay. Intrinsics are Python scalars baked from _cam0.
        # ------------------------------------------------------------------ #
        class StaticCamera:
            world_view_transform = static_world_view
            projection_matrix    = static_proj_matrix
            camera_center        = static_camera_center
            FoVx                 = baked_fovx
            FoVy                 = baked_fovy
            focal_x              = baked_focal_x
            focal_y              = baked_focal_y
            image_width          = W
            image_height         = H

        static_camera = StaticCamera()

        def copy_camera_and_target(camera, rgb):
            static_world_view.copy_(camera.world_view_transform)
            static_proj_matrix.copy_(camera.projection_matrix)
            static_camera_center.copy_(camera.camera_center)
            static_rgb.copy_(rgb)

        def get_random_batch():
            ind    = np.random.choice(len(self.data['camera']))
            camera = to_viewpoint_camera(self.data['camera'][ind])
            rgb    = self.data['rgb'][ind]
            return camera, rgb

        # ------------------------------------------------------------------ #
        # Phase 1 — Warmup (3 iterations in lazy mode).
        # Required to initialise FusedAdamW moment tensors (m, v) at fixed
        # HPU addresses before graph capture. Without this, the first
        # optimizer.step() inside the captured graph would allocate new
        # tensors, invalidating the capture.
        # ------------------------------------------------------------------ #
        self.opt.zero_grad(set_to_none=False)
        for _ in range(3):
            camera, rgb = get_random_batch()
            copy_camera_and_target(camera, rgb)

            out       = self.gaussRender(pc=self.model, camera=static_camera)
            htcore.mark_step()
            l1        = loss_utils.l1_loss(out['render'], static_rgb)
            ssim      = 1.0 - loss_utils.ssim(out['render'], static_rgb)
            loss      = (1 - self.lambda_dssim) * l1 + self.lambda_dssim * ssim
            htcore.mark_step()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad(set_to_none=False)
            htcore.mark_step()

        # ------------------------------------------------------------------ #
        # Phase 2 — Graph capture.
        # Captures: zero_grad → forward → loss → backward → optimizer.step()
        # All tensors inside are at fixed HPU addresses after warmup.
        # zero_grad(set_to_none=False) at the START of the captured region
        # ensures each replay correctly resets gradients before accumulation.
        # mark_step() is NOT needed — HPU Graphs handle synchronisation
        # implicitly.
        # NOTE: lambda_depth=0.0 so depth_loss is excluded. The boolean
        # indexing it requires (variable-length output) is incompatible with
        # HPU Graph capture.
        # ------------------------------------------------------------------ #
        g = htcore.hpu.HPUGraph()

        with htcore.hpu.graph(g):
            self.opt.zero_grad(set_to_none=False)
            static_out  = self.gaussRender(pc=self.model, camera=static_camera)
            static_l1   = loss_utils.l1_loss(static_out['render'], static_rgb)
            static_ssim = 1.0 - loss_utils.ssim(static_out['render'], static_rgb)
            static_loss = (1 - self.lambda_dssim) * static_l1 + self.lambda_dssim * static_ssim
            static_loss.backward()
            self.opt.step()

        # ------------------------------------------------------------------ #
        # Phase 3 — Replay loop.
        # Per iteration: copy new inputs → g.replay() → cadenced logging.
        # .item() is called only every i_print steps to avoid per-step
        # device→host syncs breaking graph performance.
        # on_evaluate_step() and save() run outside the graph context.
        # ------------------------------------------------------------------ #
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:

                camera, rgb = get_random_batch()
                copy_camera_and_target(camera, rgb)

                g.replay()

                self.step += 1

                # Cadenced logging — host sync every i_print steps only
                if self.step % self.i_print == 0:
                    pbar.set_description(f'loss: {static_loss.item():.3f}')

                if self.step % self.i_image == 0:
                    self.on_evaluate_step()

                if self.step != 0 and self.step % self.i_save == 0:
                    milestone = self.step // self.i_save
                    self.save(milestone)

                pbar.update(1)


if __name__ == "__main__":
    device = 'cuda'
    folder = './B075X65R3X'
    data = read_all(folder, resize_factor=0.5)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1, 3]] * len(data['rgb'])).to(device)

    points    = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    raw_points = points.random_sample(2**14)

    gaussModel = GaussModel(sh_degree=4, debug=False)
    gaussModel.create_from_pcd(pcd=raw_points)

    render_kwargs = {
        'white_bkgd':    True,
        'image_height':  256,
        'image_width':   256,
    }

    trainer = GSSTrainer(
        model=gaussModel,
        data=data,
        train_batch_size=1,
        train_num_steps=25000,
        i_image=100,
        train_lr=1e-3,
        amp=False,
        fp16=False,
        results_folder='result/test',
        render_kwargs=render_kwargs,
    )

    trainer.on_evaluate_step()
    trainer.train()
