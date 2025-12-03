"""
Multiplicação paralela local de matrizes (multi-core) usando shared memory
"""
import numpy as np
from numba import jit
import time
import multiprocessing as mp
from multiprocessing import shared_memory


@jit(nopython=True, cache=True)
def _multiply_block_range(A, B, C, linha_inicial, linha_final, tamanho_bloco):
    """
    Multiplica um range de linhas da matriz.
    
    Args:
        A: Matriz A
        B: Matriz B
        C: Matriz C (resultado)
        start_row: Linha inicial
        end_row: Linha final (exclusiva)
        block_size: Tamanho do bloco para tiling
    """
    colunas_A = A.shape[1]
    colunas_B = B.shape[1]
    
    for linha in range(linha_inicial, linha_final):
        for inicio_bloco_col in range(0, colunas_B, tamanho_bloco):
            for inicio_bloco_k in range(0, colunas_A, tamanho_bloco):
                fim_bloco_col = min(inicio_bloco_col + tamanho_bloco, colunas_B)
                fim_bloco_k = min(inicio_bloco_k + tamanho_bloco, colunas_A)
                
                for coluna in range(inicio_bloco_col, fim_bloco_col):
                    temp = 0.0
                    for k_iter in range(inicio_bloco_k, fim_bloco_k):
                        temp += A[linha, k_iter] * B[k_iter, coluna]
                    C[linha, coluna] += temp


def _worker_process(shm_A_name, shm_B_name, shm_C_name, shape_A, shape_B, 
                    linha_inicial, linha_final, tamanho_bloco):
    """
    Processo worker para multiplicação paralela.
    
    Args:
        shm_A_name: Nome da shared memory para A
        shm_B_name: Nome da shared memory para B
        shm_C_name: Nome da shared memory para C
        shape_A: Shape da matriz A
        shape_B: Shape da matriz B
        start_row: Linha inicial
        end_row: Linha final
        block_size: Tamanho do bloco
    """
    # Anexar às shared memories
    shm_A = shared_memory.SharedMemory(name=shm_A_name)
    shm_B = shared_memory.SharedMemory(name=shm_B_name)
    shm_C = shared_memory.SharedMemory(name=shm_C_name)
    
    # Criar arrays a partir das shared memories
    A = np.ndarray(shape_A, dtype=np.float64, buffer=shm_A.buf)
    B = np.ndarray(shape_B, dtype=np.float64, buffer=shm_B.buf)
    C = np.ndarray((shape_A[0], shape_B[1]), dtype=np.float64, buffer=shm_C.buf)
    
    # Executar multiplicação
    _multiply_block_range(A, B, C, linha_inicial, linha_final, tamanho_bloco)
    
    # Fechar shared memories
    shm_A.close()
    shm_B.close()
    shm_C.close()


def multiply_parallel_local(matA, matB, num_cores=None, block_size=64):
    """
    Multiplica duas matrizes usando múltiplos núcleos com shared memory.
    
    Args:
        matA: Matriz A (NumPy array)
        matB: Matriz B (NumPy array)
        num_cores: Número de núcleos a usar (None = todos disponíveis)
        block_size: Tamanho do bloco para tiling
        
    Returns:
        tuple: (matC, tempo_execucao)
    """
    # Validar dimensões
    if matA.shape[1] != matB.shape[0]:
        raise ValueError(
            f"Dimensões incompatíveis: {matA.shape} x {matB.shape}"
        )
    
    # Determinar número de cores
    if num_cores is None:
        num_cores = mp.cpu_count()
    num_cores = min(num_cores, mp.cpu_count())
    
    linhas_A, colunas_A = matA.shape
    colunas_B = matB.shape[1]
    
    # Criar shared memories
    shm_A = shared_memory.SharedMemory(create=True, size=matA.nbytes)
    shm_B = shared_memory.SharedMemory(create=True, size=matB.nbytes)
    shm_C = shared_memory.SharedMemory(create=True, size=linhas_A * colunas_B * 8)
    
    # Copiar dados para shared memory
    np_A = np.ndarray(matA.shape, dtype=np.float64, buffer=shm_A.buf)
    np_B = np.ndarray(matB.shape, dtype=np.float64, buffer=shm_B.buf)
    np_C = np.ndarray((linhas_A, colunas_B), dtype=np.float64, buffer=shm_C.buf)
    
    np_A[:] = matA[:]
    np_B[:] = matB[:]
    np_C.fill(0.0)
    
    # Dividir trabalho entre processos
    linhas_por_processo = linhas_A // num_cores
    processes = []
    
    # Medir apenas o tempo de multiplicação
    start_time = time.perf_counter()
    
    for i in range(num_cores):
        linha_inicial = i * linhas_por_processo
        if i == num_cores - 1:
            linha_final = linhas_A  # Último processo pega linhas restantes
        else:
            linha_final = (i + 1) * linhas_por_processo
        
        p = mp.Process(
            target=_worker_process,
            args=(shm_A.name, shm_B.name, shm_C.name, matA.shape, matB.shape,
                  linha_inicial, linha_final, block_size)
        )
        processes.append(p)
        p.start()
    
    # Aguardar conclusão
    for p in processes:
        p.join()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    # Copiar resultado
    result = np_C.copy()
    
    # Limpar shared memories
    shm_A.close()
    shm_B.close()
    shm_C.close()
    shm_A.unlink()
    shm_B.unlink()
    shm_C.unlink()
    
    return result, elapsed_time