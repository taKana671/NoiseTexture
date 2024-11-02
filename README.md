# NoiseTexture

This repository contains python and cython codes that can generate noise images, which can be used for texture and heightmap to visualize the terrain in 3D. 
In the python modules, numpy, and in the Cython modules, C array is mainly used. Those modules have the same functions, which return the array to be converted to an image.
Their difference is speed. See [speed comparison](#speed-comparison) result below.

# Requirements

* Cython 3.0.11
* numpy 2.1.2
* opencv-contrib-python 4.10.0.84
* opencv-python 4.10.0.84

# Environment

* Python 3.11
* Windows11

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
create_image_16bit(arr)

# change the number of lattices and the image size. The grid default is 4, size default is 256. 
maker = Perlin(grid=8, size=257)

```

A noise image will be output as png file.   
For more details of methods and parameters, please see source codes.   
When making terrain with Panda3D, an image which bit-depth is 16 is required.
(See [DeliveryCart](https://github.com/taKana671/DeliveryCart) or [MazeLand](https://github.com/taKana671/MazeLand))


![sample](https://github.com/user-attachments/assets/d8c7a581-de6b-4af6-90ad-a4d095d6a854)

# Speed ​​comparison

The execution time of each methods were measured like this.

```
maker = Voroni()
reslt = %timeit -o maker.noise2()
print(reslt.best, reslt.loops, reslt.repeat)
```

<table>
    <tr>
      <th></th>
      <th colspan="3">python</th>
      <th colspan="3">cython</th>
    </tr>
    <tr>
      <th>method</th>
      <th>best(s)</th>
      <th>loops</th>
      <th>repeat</th>
      <th>best(s)</th>
      <th>loops</th>
      <th>repeat</th>
    </tr>
    <tr>
      <td>Perlin.noise2</td>
      <td>1.210008</td>
      <td>1</td>
      <td>7</td>
      <td>0.017233</td>
      <td>100</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Perlin.noise3</td>
      <td>2.081957</td>
      <td>1</td>
      <td>7</td>
      <td>0.023179</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Perlin.wrap</td>
      <td>4.889988</td>
      <td>1</td>
      <td>7</td>
      <td>0.043762</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>FBM.noise2</td>
      <td>3.849672</td>
      <td>1</td>
      <td>7</td>
      <td>0.041291</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>FBM.wrap</td>
      <td>15.43603</td>
      <td>1</td>
      <td>7</td>
      <td>0.139114</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Cellular.noise2</td>
      <td>1.420607</td>
      <td>1</td>
      <td>7</td>
      <td>0.036839</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Cellular.noise3</td>
      <td>3.434327</td>
      <td>1</td>
      <td>7</td>
      <td>0.090029</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Cellular.noise24</td>
      <td>4.833801</td>
      <td>1</td>
      <td>7</td>
      <td>0.099891</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Periodic.noise2</td>
      <td>1.494618</td>
      <td>1</td>
      <td>7</td>
      <td>0.021754</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Periodic.noise3</td>
      <td>2.582619</td>
      <td>1</td>
      <td>7</td>
      <td>0.031351</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Voronoi.noise2</td>
      <td>1.464140</td>
      <td>1</td>
      <td>7</td>
      <td>0.097766</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>Voronoi.noise3</td>
      <td>3.533389</td>
      <td>1</td>
      <td>7</td>
      <td>0.158923</td>
      <td>10</td>
      <td>7</td>
    </tr>       
</table>
