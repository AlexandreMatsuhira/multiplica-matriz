"""
Servidor de computação distribuída usando Pyro5
Cada servidor paraleliza internamente usando multiprocessing
"""
import numpy as np
import Pyro5.api
import multiprocessing as mp
from multiprocessing import shared_memory
from numba import jit


@jit(nopython=True, cache=True)
def _multiply_block_range(A, B, C, start_row, end_row, block_size):
    """Multiplica um range de linhas da matriz."""
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
    """Processo worker para computação paralela."""
    shm_A = shared_memory.SharedMemory(name=shm_A_name)
    shm_B = shared_memory.SharedMemory(name=shm_B_name)
    shm_C = shared_memory.SharedMemory(name=shm_C_name)
    
    A = np.ndarray(shape_A, dtype=np.float64, buffer=shm_A.buf)
    B = np.ndarray(shape_B, dtype=np.float64, buffer=shm_B.buf)
    C = np.ndarray((shape_A[0], shape_B[1]), dtype=np.float64, buffer=shm_C.buf)
    
    _multiply_block_range(A, B, C, start_row, end_row, block_size)
    
    shm_A.close()
    shm_B.close()
    shm_C.close()


@Pyro5.api.expose
class ComputationServer:
    """
    Servidor que executa multiplicação de matrizes de forma paralela.
    """
    
    def __init__(self, name="server"):
        self.name = name
        self.num_cores = mp.cpu_count()
        print(f"[{self.name}] Servidor iniciado com {self.num_cores} núcleos")
    
    def get_info(self):
        """Retorna informações sobre o servidor."""
        return {
            "name": self.name,
            "num_cores": self.num_cores
        }
    
    def compute_partial_multiplication(self, A_data, B_data, start_row, end_row, 
                                      block_size=64):
        """
        Calcula multiplicação parcial de matrizes usando paralelização local.
        
        Args:
            A_data: Dados da matriz A (lista de listas)
            B_data: Dados da matriz B (lista de listas)
            start_row: Linha inicial para computar
            end_row: Linha final para computar
            block_size: Tamanho do bloco
            
        Returns:
            list: Linhas computadas da matriz resultado
        """
        # Converter para numpy arrays
        A = np.array(A_data, dtype=np.float64)
        B = np.array(B_data, dtype=np.float64)
        
        m, n = A.shape
        p = B.shape[1]
        
        # Ajustar limites
        start_row = max(0, start_row)
        end_row = min(m, end_row)
        num_rows = end_row - start_row
        
        # Se tiver poucas linhas, computar diretamente
        if num_rows <= self.num_cores:
            result = np.zeros((num_rows, p), dtype=np.float64)
            _multiply_block_range(A, B, result, start_row, end_row, block_size)
            return result.tolist()
        
        # Paralelizar entre núcleos disponíveis
        shm_A = shared_memory.SharedMemory(create=True, size=A.nbytes)
        shm_B = shared_memory.SharedMemory(create=True, size=B.nbytes)
        shm_C = shared_memory.SharedMemory(create=True, size=num_rows * p * 8)
        
        np_A = np.ndarray(A.shape, dtype=np.float64, buffer=shm_A.buf)
        np_B = np.ndarray(B.shape, dtype=np.float64, buffer=shm_B.buf)
        np_C = np.ndarray((num_rows, p), dtype=np.float64, buffer=shm_C.buf)
        
        np_A[:] = A[:]
        np_B[:] = B[:]
        np_C.fill(0.0)
        
        # Dividir trabalho entre núcleos
        rows_per_core = num_rows // self.num_cores
        processes = []
        
        for i in range(self.num_cores):
            local_start = i * rows_per_core
            if i == self.num_cores - 1:
                local_end = num_rows
            else:
                local_end = (i + 1) * rows_per_core
            
            # Ajustar para índices globais
            global_start = start_row + local_start
            global_end = start_row + local_end
            
            p = mp.Process(
                target=_worker_process,
                args=(shm_A.name, shm_B.name, shm_C.name, A.shape, B.shape,
                      global_start, global_end, block_size)
            )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        result = np_C.copy()
        
        shm_A.close()
        shm_B.close()
        shm_C.close()
        shm_A.unlink()
        shm_B.unlink()
        shm_C.unlink()
        
        return result.tolist()