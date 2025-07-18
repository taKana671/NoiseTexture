# NoiseTexture

This repository contains python and cython codes that can generate noise images, which can be used for textures and the heightmap to visualize the terrain in 3D. 
In the python modules, numpy, and in the Cython modules, C array is mainly used. Those modules have the same functions, which return the array to be converted to an image.
Their difference is speed. See [speed comparison](#speed-comparison) result below.  
I am currently trying to generate tileable noise images. See [tileable](#tileable) for noise that has become tileable.
</br>
</br>
Also submodule of  
* https://github.com/taKana671/TextureGenerator
* https://github.com/taKana671/ParticleText

# Requirements

* Cython 3.0.12
* numpy 2.2.4
* opencv-contrib-python 4.11.0.86
* opencv-python 4.11.0.86

# Environment

* Python 3.12
* Windows11

# Building Cython code

### Clone this repository.
```
git clone https://github.com/taKana671/NoiseTexture.git
```

### Build cython code.
```
cd NoiseTexture
python setup.py build_ext --inplace
```
If the error like "ModuleNotFoundError: No module named ‘distutils’" occurs, install the setuptools.
```
pip install setuptools
```

# Noise Images

### Example
```
from cynoise.perlin import PerlinNoise
# from pynoise.perlin import PerlinNoise
from output_image import output_image_8bit, output_image_16bit

maker = PerlinNoise()
arr = maker.noise3()
output_image_8bit(arr)
output_image_16bit(arr)

# change the number of lattices and the image size. The grid default is 4, size default is 256. 
maker = PerlinNoise()
arr = maker.noise3(size=512, grid=8)

```

### Noise that can be generated
A noise image is output as png file.   
For more details of methods and parameters, please see source codes.

***
#### Perlin Noise

![Image](https://github.com/user-attachments/assets/7a6ba469-236a-4e83-b5ac-e5d8f1d4d7f9)

***
#### Cellular Noise
![Image](https://github.com/user-attachments/assets/3739693c-c730-49b5-9514-f73b035a2c80)

***
#### Periodic Noise
![Image](https://github.com/user-attachments/assets/6b621b18-cd04-4501-b2a6-536a86bc22be)

***
#### Voronoi Noise
![Image](https://github.com/user-attachments/assets/049c83e9-410f-4fb7-b10e-9f4d5d44fba5)
![Image](https://github.com/user-attachments/assets/4d29f930-662e-4e55-809e-32fcc82c93b0)

***
#### Simplex Noise
![Image](https://github.com/user-attachments/assets/321264ed-4ed8-41bd-a074-43ba89df1bb0)

***
#### Value Noise
![Image](https://github.com/user-attachments/assets/5ace87b4-f5e2-4e68-9983-6869f1e34374)

***
#### Tileable
![Image](https://github.com/user-attachments/assets/ce5abe2d-c37c-4013-81f6-afdadac39b5b)
![Image](https://github.com/user-attachments/assets/ba7e60f3-303e-4f04-a136-9d81e32d2207)
![Image](https://github.com/user-attachments/assets/1965276a-e06d-457e-a6c6-9a87cd6dc04f)


# Speed ​​comparison

The execution time of each methods were measured like below.

```
maker = VoroniNoise()
result = %timeit -o maker.noise2()
print(result.best, result.loops, result.repeat)
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
      <td>PerlinNoise.noise2</td>
      <td>1.258439</td>
      <td>1</td>
      <td>7</td>
      <td>0.016928</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>PerlinNoise.noise3</td>
      <td>2.113114</td>
      <td>1</td>
      <td>7</td>
      <td>0.023845</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>PerlinNoise.noise4</td>
      <td>4.339920</td>
      <td>1</td>
      <td>7</td>
      <td>0.048453</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>PerlinNoise.fractal2</td>
      <td>5.185004</td>
      <td>1</td>
      <td>7</td>
      <td>0.047277</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>PerlinNoise.warp2_rot</td>
      <td>21.5.049</td>
      <td>1</td>
      <td>7</td>
      <td>0.167894</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>PerlinNoise.wrap2</td>
      <td>20.60505</td>
      <td>1</td>
      <td>7</td>
      <td>0.162479</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>CellularNoise.noise2</td>
      <td>1.772891</td>
      <td>1</td>
      <td>7</td>
      <td>0.034365</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>CellularNoise.noise3</td>
      <td>4.445742</td>
      <td>1</td>
      <td>7</td>
      <td>0.076830</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>CellularNoise.noise24</td>
      <td>5.562702</td>
      <td>1</td>
      <td>7</td>
      <td>0.089216</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>CellularNoise.cnoise2</td>
      <td>5.574327</td>
      <td>1</td>
      <td>7</td>
      <td>0.146625</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>CellularNoise.cnoise3</td>
      <td>15.21613</td>
      <td>1</td>
      <td>7</td>
      <td>0.330184</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>PeriodicNoise.noise2</td>
      <td>1.511534</td>
      <td>1</td>
      <td>7</td>
      <td>0.017240</td>
      <td>100</td>
      <td>7</td>
    </tr>
    <tr>
      <td>PeriodicNoise.noise3</td>
      <td>2.522443</td>
      <td>1</td>
      <td>7</td>
      <td>0.023741</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>VoronoiNoise.noise2</td>
      <td>1.782102</td>
      <td>1</td>
      <td>7</td>
      <td>0.113907</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>VoronoiNoise.noise3</td>
      <td>4.200216</td>
      <td>1</td>
      <td>7</td>
      <td>0.179295</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>VoronoiEdges.noise2</td>
      <td>16.96586</td>
      <td>1</td>
      <td>7</td>
      <td>0.195019</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>VoronoiEdges.noise3</td>
      <td>83.08460</td>
      <td>1</td>
      <td>7</td>
      <td>1.026584</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>VoronoiRoundEdges.noise2</td>
      <td>18.23457</td>
      <td>1</td>
      <td>7</td>
      <td>0.245364</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>VoronoiRoundEdges.noise3</td>
      <td>84.45329</td>
      <td>1</td>
      <td>7</td>
      <td>1.233588</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <td>SimplexNoise.noise2</td>
      <td>1.656967</td>
      <td>1</td>
      <td>7</td>
      <td>0.020974</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>SimplexNoise.noise3</td>
      <td>4.337698</td>
      <td>1</td>
      <td>7</td>
      <td>0.024148</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
      <td>SimplexNoise.fractal2</td>
      <td>6.711880</td>
      <td>1</td>
      <td>7</td>
      <td>0.065214</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <td>SimplexNoise.fractal3</td>
      <td>17.35970</td>
      <td>1</td>
      <td>7</td>
      <td>0.083178</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
    <td>ValueNoise.noise2</td>
      <td>1.128080</td>
      <td>1</td>
      <td>7</td>
      <td>0.016823</td>
      <td>100</td>
      <td>7</td>
    </tr>
    <tr>
    <td>ValueNoise.noise3</td>
      <td>1.566021</td>
      <td>1</td>
      <td>7</td>
      <td>0.022034</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
    <td>ValueNoise.grad2</td>
      <td>3.875698</td>
      <td>1</td>
      <td>7</td>
      <td>0.034712</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
    <td>ValueNoise.fractal2</td>
      <td>3.839937</td>
      <td>1</td>
      <td>7</td>
      <td>0.041812</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
    <td>ValueNoise.warp2_rot</td>
      <td>16.90730</td>
      <td>1</td>
      <td>7</td>
      <td>0.151481</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
    <td>ValueNoise.warp2</td>
      <td>15.49489</td>
      <td>1</td>
      <td>7</td>
      <td>0.142454</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
    <td>TileablePerlinNoise.tileable_noise</td>
      <td>4.531276</td>
      <td>1</td>
      <td>7</td>
      <td>0.061470</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
    <td>TileableSimplexNoise.tileable_noise</td>
      <td>8.264322</td>
      <td>1</td>
      <td>7</td>
      <td>0.052569</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <tr>
    <td>TileableVoronoiNoise.noise2</td>
      <td>3.651836</td>
      <td>1</td>
      <td>7</td>
      <td>0.136157</td>
      <td>10</td>
      <td>7</td>
    </tr>
    <td>TileableVoronoiNoise.noise3</td>
      <td>11.14920</td>
      <td>1</td>
      <td>7</td>
      <td>0.286601</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <td>TileableVoronoiEdges.noise2</td>
      <td>22.70371</td>
      <td>1</td>
      <td>7</td>
      <td>0.316223</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <td>TileableVoronoiEdges.noise3</td>
      <td>110.0342</td>
      <td>1</td>
      <td>7</td>
      <td>1.719043</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <td>TileableVoronoiRoundEdges.noise2</td>
      <td>23.40995</td>
      <td>1</td>
      <td>7</td>
      <td>0.356037</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <td>TileableVoronoiRoundEdges.noise3</td>
      <td>116.0966</td>
      <td>1</td>
      <td>7</td>
      <td>1.994004</td>
      <td>1</td>
      <td>7</td>
    </tr>
</table>
