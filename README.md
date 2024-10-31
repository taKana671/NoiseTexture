# NoiseTexture

This repository contains python and cython modules that can generate noise images, which can be used for texture and heightmap to visualize the terrain in 3D. 


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
When making terrain with Panda3D, an image which bit-depth is 16 is required.
(See [DeliveryCart](https://github.com/taKana671/DeliveryCart) or [MazeLand](https://github.com/taKana671/MazeLand))


![sample](https://github.com/user-attachments/assets/d8c7a581-de6b-4af6-90ad-a4d095d6a854)

# Speed ​​comparison

|                   |              python                                                  |        cython
|-------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------                     
| Perlin.noise2     | 1.21 s ± 18.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)  |   17.5 ms ± 178 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

