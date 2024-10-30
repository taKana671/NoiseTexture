from datetime import datetime

import cv2
import numpy as np

from perlin import Perlin
from fBm import FractionalBrownianMotion as FBM
from cellular import Cellular
from voronoi import Voronoi
from periodic import Periodic


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
    start = datetime.now()
    maker = Perlin(grid=8)
    arr = maker.wrap(rot=True)
    create_image_8bit(arr)

    print(f'It took {datetime.now() - start}.')


if __name__ == '__main__':
    run()
    # import cProfile
    # cProfile.run('run()', sort='time')

# if __name__ == '__main__':
#     from perlin import Perlin
#     from cellular import Cellular
#     from voronoi import Voronoi
#     from fBm import FractionalBrownianMotion as FBM

#     maker = FBM()
#     arr = maker.wrap()

    # maker = Voronoi()
    # arr = maker.noise3()

    # maker = Perlin()
    # arr = maker.noise2()
    # maker = Cellular(grid=8)
    # arr = maker.noise24()
    # create_image_8bit(arr)