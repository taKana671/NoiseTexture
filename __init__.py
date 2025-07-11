try:
    from .cynoise.cellular import CellularNoise
    from .cynoise.fBm import Fractal2D, Fractal3D
    from .cynoise.periodic import PeriodicNoise
    from .cynoise.perlin import PerlinNoise, TileablePerlinNoise
    from .cynoise.simplex import SimplexNoise, TileableSimplexNoise
    from .cynoise.value import ValueNoise
    from .cynoise.voronoi.voronoi import VoronoiNoise, TileableVoronoiNoise
    from .cynoise.voronoi.edges import VoronoiEdges, TileableVoronoiEdges
    from .cynoise.voronoi.rounded_edges import VoronoiRoundEdges, TileableVoronoiRoundEdges
    from .cynoise.warping import DomainWarping2D, DomainWarping3D
    print('Use cython code.')
except ImportError:
    from .pynoise.cellular import CellularNoise
    from .pynoise.fBm import Fractal2D, Fractal3D
    from .pynoise.periodic import PeriodicNoise
    from .pynoise.perlin import PerlinNoise, TileablePerlinNoise
    from .pynoise.simplex import SimplexNoise, TileableSimplexNoise
    from .pynoise.warping import DomainWarping2D, DomainWarping3D
    from .pynoise.voronoi.voronoi import VoronoiNoise, TileableVoronoiNoise
    from .pynoise.voronoi.edges import VoronoiEdges, TileableVoronoiEdges
    from .pynoise.voronoi.rounded_edges import VoronoiRoundEdges, TileableVoronoiRoundEdges
    from .pynoise.value import ValueNoise
    print('Use python code.')
