"""
Configurações globais do projeto
"""
import os
from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Criar diretórios se não existirem
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Arquivos
STATS_FILE = RESULTS_DIR / "stats.json"

# Configurações de multiplicação
DEFAULT_BLOCK_SIZE = 64  # Tamanho do bloco para tiling
BENCHMARK_ITERATIONS = 5  # Número de iterações para benchmark

# Configurações de Pyro
PYRO_HOST = "localhost"
PYRO_NS_HOST = "localhost"
PYRO_NS_PORT = 9090

# Configurações de paralelização
import multiprocessing
MAX_CORES = multiprocessing.cpu_count()
