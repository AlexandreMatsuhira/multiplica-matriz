"""
Multiplicação linear de matrizes (single-core) com Numba
"""
import numpy as np
from numba import jit
import time


@jit(nopython=True, cache=True)
def _multiply_kernel(A, B, C, tamanho_bloco):
    """
    Kernel otimizado para multiplicação de matrizes usando tiling.
    
    Args:
        A: Matriz A (m x n)
        B: Matriz B (n x p)
        C: Matriz C resultado (m x p)
        block_size: Tamanho do bloco para cache blocking
    """
    linhas_A, colunas_A = A.shape
    colunas_B = B.shape[1]
    
    # Loop tiling para melhor uso de cache]
#  C[i,j]=k=0∑n−1​A[i,k]⋅B[k,j]
    for inicio_bloco_linha in range(0, linhas_A, tamanho_bloco):
        for inicio_bloco_col in range(0, colunas_B, tamanho_bloco):
            for inicio_bloco_k in range(0, colunas_A, tamanho_bloco):
                # Limites dos blocos
                fim_bloco_linha = min(inicio_bloco_linha + tamanho_bloco, linhas_A)
                fim_bloco_col = min(inicio_bloco_col + tamanho_bloco, colunas_B)
                fim_bloco_k = min(inicio_bloco_k + tamanho_bloco, colunas_A)
                
                # Multiplicação do bloco
                for linha in range(inicio_bloco_linha, fim_bloco_linha):
                    for coluna in range(inicio_bloco_col, fim_bloco_col):
                        temp = 0.0
                        for k_iter in range(inicio_bloco_k, fim_bloco_k):
                            temp += A[linha, k_iter] * B[k_iter, coluna]
                        C[linha, coluna] += temp


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
    linhas_A, colunas_A = matA.shape
    colunas_B = matB.shape[1]
    matC = np.zeros((linhas_A, colunas_B), dtype=np.float64)
    
    # Warm-up para JIT compilation
    if matA.shape[0] > 10:
        _multiply_kernel(
            matA[:10, :10] if colunas_A >= 10 else matA[:10, :colunas_A],
            matB[:10, :10] if colunas_B >= 10 else matB[:colunas_A, :10],
            np.zeros((10, 10 if colunas_B >= 10 else colunas_B), dtype=np.float64),
            min(block_size, 10)
        )
    
    # Medir apenas o tempo de multiplicação
    start_time = time.perf_counter()
    _multiply_kernel(matA, matB, matC, block_size)
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    
    return matC, elapsed_time
