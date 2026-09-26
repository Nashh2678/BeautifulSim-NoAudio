import os

ROOT_PATH = os.path.normpath(os.path.join(__file__, '../../')) + "/"
DATA_DIR_PATH = ROOT_PATH + "data/"

WINDOW_SIZE_X = 1440
WINDOW_SIZE_Y = 960

# Max GL RENDER resolution (main.py QRSVGLWidget._ensure_render_target) -- NOT a window-size limit,
# the window itself resizes freely. A window bigger than this gets the capped image upscaled instead
# of the scene actually being drawn at the larger size (4x MSAA at 4K was the real fps killer).
MAX_RENDER_W = 1920
MAX_RENDER_H = 1080