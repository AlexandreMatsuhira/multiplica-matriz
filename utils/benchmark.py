"""
Funções para benchmark de hardware
"""
import time
import platform
import psutil
import numpy as np
from numba import jit


@jit(nopython=True)
def _integer_benchmark_kernel(n):
    """Kernel para benchmark de inteiros"""
    result = 0
    for i in range(n):
        result += i * i - i // 2
    return result


@jit(nopython=True)
def _float_benchmark_kernel(n):
    """Kernel para benchmark de ponto flutuante"""
    result = 0.0
    for i in range(n):
        result += float(i) * 1.5 - float(i) / 2.3 + np.sqrt(float(i))
    return result


def benchmark_integer_operations(iterations=10000000):
    """
    Executa benchmark de operações com inteiros.
    
    Args:
        iterations: Número de iterações
        
    Returns:
        dict: Estatísticas do benchmark
    """
    # Warm-up para JIT compilation
    _integer_benchmark_kernel(1000)
    
    start = time.perf_counter()
    result = _integer_benchmark_kernel(iterations)
    end = time.perf_counter()
    
    elapsed = end - start
    ops_per_second = iterations / elapsed if elapsed > 0 else 0
    
    return {
        "operations": iterations,
        "time_seconds": elapsed,
        "ops_per_second": ops_per_second,
        "result": int(result)
    }


def benchmark_float_operations(iterations=10000000):
    """
    Executa benchmark de operações com ponto flutuante.
    
    Args:
        iterations: Número de iterações
        
    Returns:
        dict: Estatísticas do benchmark
    """
    # Warm-up para JIT compilation
    _float_benchmark_kernel(1000)
    
    start = time.perf_counter()
    result = _float_benchmark_kernel(iterations)
    end = time.perf_counter()
    
    elapsed = end - start
    ops_per_second = iterations / elapsed if elapsed > 0 else 0
    
    return {
        "operations": iterations,
        "time_seconds": elapsed,
        "ops_per_second": ops_per_second,
        "result": float(result)
    }


def get_system_info():
    """
    Coleta informações do sistema.
    
    Returns:
        dict: Informações do sistema
    """
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info.get('brand_raw', 'Unknown')
        cpu_freq = cpu_info.get('hz_actual_friendly', 'Unknown')
    except:
        cpu_name = platform.processor()
        cpu_freq = "Unknown"
    
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": cpu_name,
        "cpu_frequency": cpu_freq,
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }


def run_benchmark(iterations_int=10000000, iterations_float=10000000):
    """
    Executa benchmark completo do sistema.
    
    Args:
        iterations_int: Iterações para benchmark de inteiros
        iterations_float: Iterações para benchmark de floats
        
    Returns:
        dict: Resultados completos do benchmark
    """
    print("Executando benchmark do sistema...")
    print("=" * 60)
    
    system_info = get_system_info()
    print(f"Sistema: {system_info['platform']} {system_info['platform_release']}")
    print(f"Processador: {system_info['processor']}")
    print(f"Cores: {system_info['cpu_cores']} | Threads: {system_info['cpu_threads']}")
    print(f"RAM: {system_info['ram_total_gb']} GB")
    print("=" * 60)
    
    print("\nBenchmark de operações com inteiros...")
    int_bench = benchmark_integer_operations(iterations_int)
    print(f"  Tempo: {int_bench['time_seconds']:.4f}s")
    print(f"  Ops/seg: {int_bench['ops_per_second']:.2e}")
    
    print("\nBenchmark de operações com ponto flutuante...")
    float_bench = benchmark_float_operations(iterations_float)
    print(f"  Tempo: {float_bench['time_seconds']:.4f}s")
    print(f"  Ops/seg: {float_bench['ops_per_second']:.2e}")
    
    print("=" * 60)
    
    return {
        "system_info": system_info,
        "integer_benchmark": int_bench,
        "float_benchmark": float_bench
    }
