"""
Servidor de computação distribuída usando Pyro5
CORRIGIDO: Mapeamento correto de Shared Memory para evitar sobrescrita
"""
import numpy as np
from datetime import datetime
import Pyro5.api
import multiprocessing as mp
from multiprocessing import shared_memory
from numba import jit

# ================================
#   MULTIPLICAÇÃO COM OFFSET
# ================================
@jit(nopython=True, cache=True)
def _multiply_block_range(A, B, C, start_row, end_row, block_size, offset):
    """
    Multiplica o intervalo global [start_row, end_row).
    O resultado é gravado em C na posição relativa (i - offset).
    """
    n = A.shape[1]
    p = B.shape[1]
    
    # Loop nas linhas globais da matriz A
    for i in range(start_row, end_row):
        # Índice local dentro do buffer de resultado deste servidor
        local_i = i - offset 
        
        for j0 in range(0, p, block_size):
            for k0 in range(0, n, block_size):
                j_max = min(j0 + block_size, p)
                k_max = min(k0 + block_size, n)
                
                for j in range(j0, j_max):
                    temp = 0.0
                    for k in range(k0, k_max):
                        # Acesso a A deve ser pelo índice local (já que A é um slice)
                        # Se A fosse a matriz completa, seria A[i, k]
                        # Como A é slice [start_row:end_row], o índice 0 corresponde a start_row
                        # Portanto, acessamos A[local_i, k]
                        temp += A[local_i, k] * B[k, j]
                    
                    # Acumula no buffer local
                    C[local_i, j] += temp


# ================================
#   WORKER PROCESS
# ================================
def _worker_process(shm_A_name, shm_B_name, shm_C_name, shape_A, shape_B,
                    worker_start, worker_end, server_total_rows, server_global_start, block_size):
    """
    Processo worker.
    
    Args:
        worker_start, worker_end: Intervalo de linhas globais que ESTE worker calcula.
        server_total_rows: Quantidade total de linhas que este SERVIDOR está processando (para mapear o buffer C inteiro).
        server_global_start: A linha global onde começa o bloco deste SERVIDOR (para calcular o offset).
    """
    # Conectar à memória compartilhada existente
    shm_A = shared_memory.SharedMemory(name=shm_A_name)
    shm_B = shared_memory.SharedMemory(name=shm_B_name)
    shm_C = shared_memory.SharedMemory(name=shm_C_name)
    
    # Recriar arrays numpy a partir dos buffers
    A = np.ndarray(shape_A, dtype=np.float64, buffer=shm_A.buf)
    B = np.ndarray(shape_B, dtype=np.float64, buffer=shm_B.buf)

    # ### CORREÇÃO AQUI ###
    # Mapeamos o C com o tamanho TOTAL de linhas designadas a este servidor,
    # e não apenas as linhas deste worker. Assim todos compartilham a mesma visão da memória.
    C = np.ndarray((server_total_rows, shape_B[1]), dtype=np.float64, buffer=shm_C.buf)
    
    # Executar multiplicação
    # O offset passado é o início global do servidor. 
    # Ex: Servidor pega linhas 100-200. Worker pega 100-150.
    # Linha 100 global - 100 offset = índice 0 no buffer C.
    _multiply_block_range(A, B, C, worker_start, worker_end, block_size, offset=server_global_start)
    
    # Fechar conexões (não dar unlink aqui, pois o pai fará isso)
    shm_A.close()
    shm_B.close()
    shm_C.close()


# ================================
#   SERVIDOR PYRO
# ================================
@Pyro5.api.expose
class ComputationServer:
    
    def __init__(self, name="server"):
        self.name = name
        self.num_cores = mp.cpu_count()
        print(f"[{self.name}] Servidor iniciado com {self.num_cores} núcleos")
    
    def get_info(self):
        return {
            "name": self.name,
            "num_cores": self.num_cores
        }
    
    def _ensure_bytes(self, obj):
        if isinstance(obj, dict) and 'data' in obj and 'encoding' in obj:
            if obj['encoding'] == 'base64':
                import base64
                return base64.b64decode(obj['data'])
        return obj

    def compute_partial_multiplication(self, A_bytes, B_bytes, A_shape, B_shape, dtype_str, 
                                      start_row, end_row, block_size=64):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Recebido pedido de cálculo: linhas {start_row} a {end_row}")
        
        # Garantir que recebemos bytes (decodificar se Serpent enviou dict)
        A_bytes = self._ensure_bytes(A_bytes)
        B_bytes = self._ensure_bytes(B_bytes)
        
        # Reconstruir arrays numpy a partir dos bytes
        dtype = np.dtype(dtype_str)
        A = np.frombuffer(A_bytes, dtype=dtype).reshape(A_shape)
        B = np.frombuffer(B_bytes, dtype=dtype).reshape(B_shape)
        
        m, n = A.shape
        p = B.shape[1]

        # Garantir limites válidos
        start_row = max(0, start_row)
        # Como A é um slice, o número de linhas disponíveis é m.
        # O intervalo solicitado [start_row, end_row) não pode exceder m linhas.
        if (end_row - start_row) > m:
            end_row = start_row + m
            
        num_rows_server = end_row - start_row
        
        if num_rows_server <= 0:
            return b""

        # -----------------------------------------
        #  CASO POUCAS LINHAS → NÃO USA SHM
        # -----------------------------------------
        if num_rows_server <= self.num_cores:
            result = np.zeros((num_rows_server, p), dtype=np.float64)
            _multiply_block_range(A, B, result, start_row, end_row, block_size, offset=start_row)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cálculo finalizado (local).")
            return result.tobytes()
        
        # -----------------------------------------
        #       SHARED MEMORY (CORRIGIDO)
        # -----------------------------------------
        # Criar memória compartilhada
        try:
            shm_A = shared_memory.SharedMemory(create=True, size=A.nbytes)
            shm_B = shared_memory.SharedMemory(create=True, size=B.nbytes)
            # Tamanho de C é apenas para as linhas deste servidor
            shm_C = shared_memory.SharedMemory(create=True, size=num_rows_server * p * 8) 
            
            # Popular buffers iniciais
            np_A = np.ndarray(A.shape, dtype=np.float64, buffer=shm_A.buf)
            np_B = np.ndarray(B.shape, dtype=np.float64, buffer=shm_B.buf)
            np_C = np.ndarray((num_rows_server, p), dtype=np.float64, buffer=shm_C.buf)
            
            np_A[:] = A[:]
            np_B[:] = B[:]
            np_C.fill(0.0) # Limpar buffer de saída
            
            # Dividir trabalho entre os núcleos locais
            rows_per_core = num_rows_server // self.num_cores
            processes = []
            
            for i in range(self.num_cores):
                # Índices relativos ao pedaço do servidor
                local_start = i * rows_per_core
                # O último pega o resto
                local_end = num_rows_server if i == self.num_cores - 1 else (i + 1) * rows_per_core
                
                # Converter para índices globais (Matriz A original)
                worker_global_start = start_row + local_start
                worker_global_end = start_row + local_end
                
                p_worker = mp.Process(
                    target=_worker_process,
                    args=(
                        shm_A.name, shm_B.name, shm_C.name, 
                        A.shape, B.shape,
                        worker_global_start, # Onde este worker começa (global)
                        worker_global_end,   # Onde este worker termina (global)
                        num_rows_server,     # Altura total do buffer C local
                        start_row,           # Offset global deste servidor (para subtração)
                        block_size
                    )
                )
                processes.append(p_worker)
                p_worker.start()
            
            for p_worker in processes:
                p_worker.join()
            
            # Copiar resultado antes de destruir a memória
            result = np_C.copy()
        
        finally:
            # Limpeza garantida
            try:
                shm_A.close(); shm_A.unlink()
                shm_B.close(); shm_B.unlink()
                shm_C.close(); shm_C.unlink()
            except:
                pass
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Cálculo finalizado (shared memory).")
        return result.tobytes()