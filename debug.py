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


def construct_camera() -> Camera:
    # T = np.array(
    #     [
    #         [
    #             0.6500730420335082,
    #             0.39120341784159907,
    #             -0.6514329788169707,
    #             -2.360061110834771,
    #         ],
    #         [
    #             -0.030078509714791275,
    #             -0.8433737543926951,
    #             -0.5364848494178629,
    #             -1.8781563714402996,
    #         ],
    #         [
    #             -0.7592761837925891,
    #             0.3683484712478404,
    #             -0.5364877262852722,
    #             -2.017242989985334,
    #         ],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )

    R = np.array(
        [
            [0.00355187, -0.32231082, 0.94662723],
            [-0.04210215, 0.94574562, 0.32216863],
            [-0.999107, -0.04099934, -0.01021081],
        ]
    )
    t = np.array([0.0529294, -0.14028193, 3.80562404])

    fx = 1384.7822619779372
    fy = 1383.4244140912685

    # fovy = focal2fov(fy, opt.height)
    # fovx = focal2fov(fx, opt.width)
    fovx = 0.7404945507637375
    fovy = 1.2093728177190548

    cam = Camera(
        colmap_id=0,
        R=R,
        T=t,
        FoVx=fovx,
        FoVy=fovy,
        image=torch.zeros([3, opt.height, opt.width]),
        gt_alpha_mask=None,
        image_name="test.jpg",
        uid=0,
    )
    return cam


opt = CONFIG()
gs_model = GaussianModel(opt.sh_degree)
# gui = GaussianSplattingGUI(opt, gs_model)

# gui.render()
gs_model.load_ply(opt.ply_path)

view_camera = construct_camera()

bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


outputs = render(view_camera, gs_model, opt, background)

img = outputs["render"].permute(1, 2, 0) * 255

img = img.detach().cpu().numpy()

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imwrite("./debug.png", img)
