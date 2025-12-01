"""
Módulos de multiplicação de matrizes
"""
from .linear import multiply_linear
from .parallel_local import multiply_parallel_local
from .parallel_distributed import multiply_parallel_distributed

__all__ = [
    'multiply_linear',
    'multiply_parallel_local',
    'multiply_parallel_distributed'
]