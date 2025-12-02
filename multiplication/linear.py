"""
Multiplicação linear de matrizes (single-core) com Numba
"""
import numpy as np
from numba import jit
import time


@jit(nopython=True, cache=True)
def _multiply_kernel(A, B, C, block_size):
    """
    Kernel otimizado para multiplicação de matrizes usando tiling.
    
    Args:
        A: Matriz A (m x n)
        B: Matriz B (n x p)
        C: Matriz C resultado (m x p)
        block_size: Tamanho do bloco para cache blocking
    """
    m, n = A.shape
    p = B.shape[1]
    
    # Loop tiling para melhor uso de cache]
#  C[i,j]=k=0∑n−1​A[i,k]⋅B[k,j]
    for i0 in range(0, m, block_size):
        for j0 in range(0, p, block_size):
            for k0 in range(0, n, block_size):
                # Limites dos blocos
                i_max = min(i0 + block_size, m)
                j_max = min(j0 + block_size, p)
                k_max = min(k0 + block_size, n)
                
                # Multiplicação do bloco
                for i in range(i0, i_max):
                    for j in range(j0, j_max):
                        temp = 0.0
                        for k in range(k0, k_max):
                            temp += A[i, k] * B[k, j]
                        C[i, j] += temp


def multiply_linear(matA, matB, block_size=64):
    """
    Multiplica duas matrizes de forma linear (single-core).
    
    Args:
        matA: Matriz A (NumPy array)
        matB: Matriz B (NumPy array)
        block_size: Tamanho do bloco para tiling
        
    Returns:
        tuple: (matC, tempo_execucao)
    """
    # Validar dimensões
    if matA.shape[1] != matB.shape[0]:
        raise ValueError(
            f"Dimensões incompatíveis: {matA.shape} x {matB.shape}"
        )
    
    # Inicializar matriz resultado
    m, n = matA.shape
    p = matB.shape[1]
    matC = np.zeros((m, p), dtype=np.float64)
    
    # Warm-up para JIT compilation
    if matA.shape[0] > 10:
        _multiply_kernel(
            matA[:10, :10] if n >= 10 else matA[:10, :n],
            matB[:10, :10] if p >= 10 else matB[:n, :10],
            np.zeros((10, 10 if p >= 10 else p), dtype=np.float64),
            min(block_size, 10)
        )
    
    # Medir apenas o tempo de multiplicação
    start_time = time.perf_counter()
    _multiply_kernel(matA, matB, matC, block_size)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    
    return matC, elapsed_time
