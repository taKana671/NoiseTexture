# NoiseTexture

This repository contains python and cython modules that can generate noise images, which can be used for heightmap to visualize the terrain in 3D. 


# Building Cython code

```
python setup.py build_ext --inplace
```

# Example

```
from cynoise.perlin import Perlin
# from pynoise.perlin import Perlin
from create_image import create_image_8bit, create_image_16bit

maker = Perlin()
arr = maker.pnoise3()
create_image_8bit(arr)
# create_image_16bit(arr)
```
When making terrain with Panda3D, image which bit-depth is 16 is required.
(See [DeliveryCart](https://github.com/taKana671/DeliveryCart) or [MazeLand](https://github.com/taKana671/MazeLand))

