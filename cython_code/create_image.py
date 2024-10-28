# from py_perlin import Perlin
from perlin import Perlin
from fBm import FractionalBrownianMotion as FBM
from cellular import Cellular
from voronoi import Voronoi
from periodic import Periodic



from datetime import datetime

import cv2
import numpy as np


def create_image_8bit(arr, ext='png'):
    now = datetime.now()
    file_name = f'image_{now.strftime("%Y%m%d%H%M%S")}.{ext}'

    arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
    cv2.imwrite(file_name, arr)


def create_image_16bit(arr):
    now = datetime.now()
    file_name = f'image_{now.strftime("%Y%m%d%H%M%S")}.png'

    arr = np.clip(arr * 65535, a_min=0, a_max=65535).astype(np.uint16)
    cv2.imwrite(file_name, arr)


def run():
    print('called!')
    start = datetime.now()
    maker = Periodic()
    arr = maker.noise2()
    create_image_8bit(arr)

    print(f'It took {datetime.now() - start}.')


if __name__ == '__main__':
    run()
    # import cProfile
    # cProfile.run('run()', sort='time')

# (py311env) C:\Users\Kanae\Desktop\py311env\NoiseTexture\cython_code>python create_image.py
# It took 0:00:02.343739.

# (py311env) C:\Users\Kanae\Desktop\py311env\NoiseTexture\cython_code>python create_image.py
# It took 0:00:05.486413.