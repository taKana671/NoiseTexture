try:
    from .cynoise.cellular import CellularNoise
    from .cynoise.fBm import Fractal2D, Fractal3D
    from .cynoise.periodic import PeriodicNoise
    from .cynoise.perlin import PerlinNoise, TileablePerlinNoise
    from .cynoise.simplex import SimplexNoise, TileableSimplexNoise
    from .cynoise.value import ValueNoise
    from .cynoise.voronoi import VoronoiNoise
    from .cynoise.warping import DomainWarping2D, DomainWarping3D
    print('Use cython code.')
except ImportError:
    from .pynoise.cellular import CellularNoise
    from .pynoise.fBm import Fractal2D, Fractal3D
    from .pynoise.periodic import PeriodicNoise
    from .pynoise.perlin import PerlinNoise, TileablePerlinNoise
    from .pynoise.simplex import SimplexNoise, TileableSimplexNoise
    from .pynoise.value import ValueNoise
    from .pynoise.voronoi import VoronoiNoise
    from .pynoise.warping import DomainWarping2D, DomainWarping3D
    print('Use python code.')