from configs import CONFIG
from viewer.window import GaussianSplattingGUI
import sys

from engine.scene.gaussian_model import GaussianModel


sys.path.append("./engine")


opt = CONFIG()
gs_model = GaussianModel(opt.sh_degree)
gui = GaussianSplattingGUI(opt, gs_model)


gui.render()
