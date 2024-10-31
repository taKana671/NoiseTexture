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


if __name__ == '__main__':
    from pynoise.cellular import Cellular
    maker = Cellular()
    arr = maker.noise3()
    create_image_8bit(arr)