from configs import CONFIG
from viewer.window import GaussianSplattingGUI
import sys

from engine.scene.gaussian_model import GaussianModel
from engine.scene.cameras import Camera

import numpy as np
from engine.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import torch
from engine.gaussian_renderer import render
import cv2
import math
import os
from glob import glob
from tqdm import tqdm


def load_camera(view_dir, height=1080, width=1920):
    R = view_dir[:3, :3]
    t = view_dir[:3, 3]

    fov = 60
    ss = math.pi / 180.0
    fovy = fov * ss

    fy = fov2focal(fovy, height)
    fovx = focal2fov(fy, width)

    cam = Camera(
        colmap_id=0,
        R=R,
        T=t,
        FoVx=fovx,
        FoVy=fovy,
        image=torch.zeros([3, height, width]),
        gt_alpha_mask=None,
        image_name=None,
        uid=0,
    )

    return cam


def render_ply(view_dir, out_path, model, opt, bg):
    os.makedirs(out_path, exist_ok=True)

    views = sorted(glob(os.path.join(view_dir, "*.npy")))

    for i, v in tqdm(enumerate(views)):
        T = np.load(v)

        cam = load_camera(T)
        outputs = render(cam, model, opt, bg)

        img = outputs["render"].permute(1, 2, 0) * 255

        img = img.detach().cpu().numpy()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out_name = os.path.join(out_path, f"{i:06d}.png")
        cv2.imwrite(out_name, img)


opt = CONFIG()
gs_model = GaussianModel(opt.sh_degree)

root_dir = "/home/swh/dataset/3d_gaussian/dataset/auro_6_8/sucai6.8/results"
views_dir = "/home/swh/dataset/3d_gaussian/dataset/auro_6_8/sucai6.8/new_views"
out_path = "/home/swh/dataset/3d_gaussian/dataset/auro_6_8/sucai6.8/render_raw"

iter_name = "iteration_7000"
bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

for i in range(1, 30):
    dir_name = f"frame_{i:02d}"
    ply_file = os.path.join(
        root_dir, dir_name, "point_cloud", iter_name, "point_cloud.ply"
    )
    gs_model.load_ply(ply_file)

    out_dir = os.path.join(out_path, dir_name)

    render_ply(views_dir, out_dir, gs_model, opt, background)

    # break
