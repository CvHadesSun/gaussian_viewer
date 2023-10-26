import dearpygui.dearpygui as dpg
import numpy as np
import math

from scipy.spatial.transform import Rotation as R

from .camera import OrbitCamera
import torch


from engine.scene.cameras import Camera

from engine.gaussian_renderer import render

import cv2

from engine.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height

        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius, fovy=opt.fovy)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.bg_color = background

        self.render_buffer = np.zeros((self.height, self.width, 3), dtype=np.float32)

        self.update_camera = True

        self.dynamic_resolution = True

        self.debug = opt.debug

        self.engine = gaussian_model

        print("loading model file...")

        self.engine.load_ply(self.opt.ply_path)

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()

        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]

        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.width,
                self.height,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        with dpg.window(tag="_primary_window", width=self.width, height=self.height):
            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300):
            #     # button theme
            #     with dpg.theme() as theme_button:
            #         with dpg.theme_component(dpg.mvButton):
            #             dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
            #             dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
            #             dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
            #             dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            #             dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):
                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(
                        label="dynamic resolution",
                        default_value=self.dynamic_resolution,
                        callback=callback_set_dynamic_resolution,
                    )
                    dpg.add_text(f"{self.width}x{self.height}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.update_camera = True

                dpg.add_combo(
                    ("image", "depth"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(
                        app_data[:3], dtype=torch.float32
                    )  # only need RGB in [0, 1]
                    self.update_camera = True

                dpg.add_color_edit(
                    (255, 255, 255),
                    label="Background Color",
                    width=200,
                    tag="_color_editor",
                    no_alpha=True,
                    callback=callback_change_bg,
                )

                # fov slider

                def callback_set_fovy(sender, app_data):
                    self.camera.fovy = app_data
                    self.update_camera = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=self.camera.fovy,
                    callback=callback_set_fovy,
                )

                # dt_gamma slider

                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.update_camera = True

                dpg.add_slider_float(
                    label="dt_gamma",
                    min_value=0,
                    max_value=0.1,
                    format="%.5f",
                    default_value=self.opt.dt_gamma,
                    callback=callback_set_dt_gamma,
                )

                ##
                # dpg.add_separator()
                # dpg.add_text("Axis-aligned bounding box:")

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")

        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.camera.orbit(dx, dy)
            self.update_camera = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.camera.scale(delta)
            self.update_camera = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.camera.pan(dx, dy)
            self.update_camera = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )
        dpg.create_viewport(
            title="Gaussian-Splatting",
            width=self.width,
            height=self.height,
            resizable=False,
        )
        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            cam = self.construct_camera()
            self.fetch_data(cam)
            dpg.render_dearpygui_frame()

    def construct_camera(
        self,
    ) -> Camera:
        R = self.camera.pose[:3, :3]
        t = self.camera.pose[:3, 3]

        T = np.array(
            [
                [
                    0.6500730420335082,
                    0.39120341784159907,
                    -0.6514329788169707,
                    -2.360061110834771,
                ],
                [
                    -0.030078509714791275,
                    -0.8433737543926951,
                    -0.5364848494178629,
                    -1.8781563714402996,
                ],
                [
                    -0.7592761837925891,
                    0.3683484712478404,
                    -0.5364877262852722,
                    -2.017242989985334,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        R = T[:3, :3]
        t = T[:3, 3]

        fx = 1384.7822619779372
        fy = 1383.4244140912685

        fovy = focal2fov(fy, self.height)
        fovx = focal2fov(fx, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name="test.jpg",
            uid=0,
        )
        return cam

    def fetch_data(self, view_camera):
        outputs = render(view_camera, self.engine, self.opt, self.bg_color)

        img = outputs["render"].permute(1, 2, 0) * 255  #

        img = img.detach().cpu().numpy()

        self.render_buffer = img
        cv2.imwrite("./test.png", img)

        # if self.update_camera:
        #     self.prepare_buffer()
