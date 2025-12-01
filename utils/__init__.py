"""
Utilitários para o projeto de multiplicação de matrizes
"""
from .matrix_io import load_matrix, save_matrix
from .benchmark import run_benchmark
from .stats import save_stats, load_stats, append_stats

__all__ = [
    'load_matrix',
    'save_matrix',
    'run_benchmark',
    'save_stats',
    'load_stats',
    'append_stats'
]