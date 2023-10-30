import dearpygui.dearpygui as dpg
import numpy as np
import math


from .camera import OrbitCamera
import torch


from engine.scene.cameras import Camera

from engine.gaussian_renderer import render

import cv2
import sys

from engine.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height

        self.window_width = opt.window_width
        self.window_height = opt.window_height

        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.bg_color = background

        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)

        self.update_camera = True

        self.dynamic_resolution = True

        self.debug = opt.debug

        self.engine = gaussian_model

        self.load_model = False

        # print("loading model file...")

        # self.engine.load_ply(self.opt.ply_path)

        # print("loading model file done.")

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

        with dpg.window(
            tag="_primary_window", width=self.window_width, height=self.window_height
        ):
            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300):
            #     # button theme
            # with dpg.theme() as theme_button:
            #     with dpg.theme_component(dpg.mvButton):
            #         dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
            #         dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
            #         dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
            #         dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            #         dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

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

                def callback(sender, app_data, user_data):
                    # print("Sender: ", sender)
                    # print("App Data: ", app_data)
                    self.load_model = False

                    self.ply_file = app_data["selections"]["point_cloud.ply"]

                    # if not self.load_model:
                    print("loading model file...")
                    self.engine.load_ply(self.ply_file)
                    print("loading model file done.")
                    self.load_model = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback,
                    id="file_dialog_id",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension(".*")
                    dpg.add_file_extension("", color=(150, 255, 150, 255))
                    dpg.add_file_extension(
                        "Ply (*.ply){.ply}", color=(0, 255, 255, 255)
                    )
                dpg.add_button(
                    label="File Selector",
                    callback=lambda: dpg.show_item("file_dialog_id"),
                )

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
            title="Gaussian-Splatting-Viewer",
            width=self.window_width,
            height=self.window_height,
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
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()

    def construct_camera(
        self,
    ) -> Camera:
        R = self.camera.opt_pose[:3, :3]
        t = self.camera.opt_pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        return cam

    def fetch_data(self, view_camera):
        outputs = render(view_camera, self.engine, self.opt, self.bg_color)

        img = outputs["render"].permute(1, 2, 0)  #

        img = img.detach().cpu().numpy().reshape(-1)

        self.render_buffer = img

        dpg.set_value("_texture", self.render_buffer)
