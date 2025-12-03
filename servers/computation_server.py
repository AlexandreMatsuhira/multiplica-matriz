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
def _multiply_block_range(A, B, C, linha_global_inicial, linha_global_final, tamanho_bloco, offset):
    """
    Multiplica o intervalo global [linha_global_inicial, linha_global_final).
    O resultado é gravado em C na posição relativa (i - offset).
    """
    colunas_A = A.shape[1]
    colunas_B = B.shape[1]
    
    # Loop nas linhas globais da matriz A
    for linha_global in range(linha_global_inicial, linha_global_final):
        # Índice local dentro do buffer de resultado deste servidor
        linha_local = linha_global - offset 
        
        for inicio_bloco_col in range(0, colunas_B, tamanho_bloco):
            for inicio_bloco_k in range(0, colunas_A, tamanho_bloco):
                fim_bloco_col = min(inicio_bloco_col + tamanho_bloco, colunas_B)
                fim_bloco_k = min(inicio_bloco_k + tamanho_bloco, colunas_A)
                
                for coluna in range(inicio_bloco_col, fim_bloco_col):
                    temp = 0.0
                    for k_iter in range(inicio_bloco_k, fim_bloco_k):
                        # Acesso a A deve ser pelo índice local (já que A é um slice)
                        # Se A fosse a matriz completa, seria A[i, k]
                        # Como A é slice [linha_global_inicial:linha_global_final], o índice 0 corresponde a linha_global_inicial
                        # Portanto, acessamos A[linha_local, k]
                        temp += A[linha_local, k_iter] * B[k_iter, coluna]
                    
                    # Acumula no buffer local
                    C[linha_local, coluna] += temp


# ================================
#   WORKER PROCESS
# ================================
def _worker_process(shm_A_name, shm_B_name, shm_C_name, shape_A, shape_B,
                    worker_start, worker_end, server_total_rows, server_global_start, tamanho_bloco):
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
    _multiply_block_range(A, B, C, worker_start, worker_end, tamanho_bloco, offset=server_global_start)
    
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
                                      linha_global_inicial, linha_global_final, tamanho_bloco=64):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Recebido pedido de cálculo: linhas {linha_global_inicial} a {linha_global_final}")
        
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
        linha_global_inicial = max(0, linha_global_inicial)
        # Como A é um slice, o número de linhas disponíveis é m.
        # O intervalo solicitado [linha_global_inicial, linha_global_final) não pode exceder m linhas.
        if (linha_global_final - linha_global_inicial) > m:
            linha_global_final = linha_global_inicial + m
            
        num_rows_server = linha_global_final - linha_global_inicial
        
        if num_rows_server <= 0:
            return b""

        # -----------------------------------------
        #  CASO POUCAS LINHAS → NÃO USA SHM
        # -----------------------------------------
        if num_rows_server <= self.num_cores:
            result = np.zeros((num_rows_server, p), dtype=np.float64)
            _multiply_block_range(A, B, result, linha_global_inicial, linha_global_final, tamanho_bloco, offset=linha_global_inicial)
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
                worker_global_start = linha_global_inicial + local_start
                worker_global_end = linha_global_inicial + local_end
                
                p_worker = mp.Process(
                    target=_worker_process,
                    args=(
                        shm_A.name, shm_B.name, shm_C.name, 
                        A.shape, B.shape,
                        worker_global_start, # Onde este worker começa (global)
                        worker_global_end,   # Onde este worker termina (global)
                        num_rows_server,     # Altura total do buffer C local
                        linha_global_inicial,           # Offset global deste servidor (para subtração)
                        tamanho_bloco
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