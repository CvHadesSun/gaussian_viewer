from configs import CONFIG
from viewer.window import GaussianSplattingGUI


from engine.scene.gaussian_model import GaussianModel


opt = CONFIG()
gs_model = GaussianModel(opt.sh_degree)
gui = GaussianSplattingGUI(opt, gs_model)

gui.render()
