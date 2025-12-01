"""
Multiplicação paralela local de matrizes (multi-core) usando shared memory
"""
import numpy as np
from numba import jit
import time
import multiprocessing as mp
from multiprocessing import shared_memory


@jit(nopython=True, cache=True)
def _multiply_block_range(A, B, C, start_row, end_row, block_size):
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
    n = A.shape[1]
    p = B.shape[1]
    
    for i in range(start_row, end_row):
        for j0 in range(0, p, block_size):
            for k0 in range(0, n, block_size):
                j_max = min(j0 + block_size, p)
                k_max = min(k0 + block_size, n)
                
                for j in range(j0, j_max):
                    temp = 0.0
                    for k in range(k0, k_max):
                        temp += A[i, k] * B[k, j]
                    C[i, j] += temp


def _worker_process(shm_A_name, shm_B_name, shm_C_name, shape_A, shape_B, 
                    start_row, end_row, block_size):
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
    _multiply_block_range(A, B, C, start_row, end_row, block_size)
    
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
    
    m, n = matA.shape
    p = matB.shape[1]
    
    # Criar shared memories
    shm_A = shared_memory.SharedMemory(create=True, size=matA.nbytes)
    shm_B = shared_memory.SharedMemory(create=True, size=matB.nbytes)
    shm_C = shared_memory.SharedMemory(create=True, size=m * p * 8)
    
    # Copiar dados para shared memory
    np_A = np.ndarray(matA.shape, dtype=np.float64, buffer=shm_A.buf)
    np_B = np.ndarray(matB.shape, dtype=np.float64, buffer=shm_B.buf)
    np_C = np.ndarray((m, p), dtype=np.float64, buffer=shm_C.buf)
    
    np_A[:] = matA[:]
    np_B[:] = matB[:]
    np_C.fill(0.0)
    
    # Dividir trabalho entre processos
    rows_per_process = m // num_cores
    processes = []
    
    # Medir apenas o tempo de multiplicação
    start_time = time.perf_counter()
    
    for i in range(num_cores):
        start_row = i * rows_per_process
        if i == num_cores - 1:
            end_row = m  # Último processo pega linhas restantes
        else:
            end_row = (i + 1) * rows_per_process
        
        p = mp.Process(
            target=_worker_process,
            args=(shm_A.name, shm_B.name, shm_C.name, matA.shape, matB.shape,
                  start_row, end_row, block_size)
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