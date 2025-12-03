"""
Multiplicação paralela distribuída usando Pyro5
Cliente
"""
import numpy as np
import time
import Pyro5.api
import concurrent.futures

def multiply_parallel_distributed(matA, matB, server_uris, block_size=64):
    """
    Multiplica duas matrizes usando servidores distribuídos.
    """
    # Validar dimensões
    if matA.shape[1] != matB.shape[0]:
        raise ValueError(
            f"Dimensões incompatíveis: {matA.shape} x {matB.shape}"
        )
    
    m, n = matA.shape
    p = matB.shape[1]
    
    # Conectar aos servidores
    servers_info = []
    
    print(f"\nConectando a {len(server_uris)} servidores...")
    for uri in server_uris:
        try:
            with Pyro5.api.Proxy(uri) as server:
                info = server.get_info()
                servers_info.append(info)
                print(f"  ✓ {info['name']}: {info['num_cores']} núcleos")
        except Exception as e:
            print(f"  ✗ Erro ao conectar a {uri}: {e}")
    
    if not servers_info:
        raise RuntimeError("Nenhum servidor disponível")
    
    valid_server_count = len(servers_info)
    
    # Converter matrizes para bytes (para transmissão eficiente Pyro)
    print("Serializando matrizes...")
    # Não convertemos A inteiro para bytes aqui, pois vamos fatiar.
    # Mas B vai inteiro para todos.
    B_bytes = matB.tobytes()
    dtype_str = str(matA.dtype)
    
    # Dividir trabalho (linhas de A) entre servidores disponíveis
    rows_per_server = m // valid_server_count
    tasks = []
    
    # Recalcular URIs baseados apenas nos que conectaram com sucesso
    active_uris = server_uris[:valid_server_count] 

    for i in range(valid_server_count):
        start_row = i * rows_per_server
        if i == valid_server_count - 1:
            end_row = m  # Último servidor pega o resto
        else:
            end_row = (i + 1) * rows_per_server
        
        tasks.append({
            'server_uri': active_uris[i],
            'start_row': start_row,
            'end_row': end_row,
            'rows': end_row - start_row
        })
    
    print(f"\nDistribuindo {m} linhas entre {valid_server_count} servidores...")
    for i, task in enumerate(tasks):
        print(f"  Servidor {i+1}: linhas {task['start_row']}-{task['end_row']} "
              f"({task['rows']} linhas)")
    
    # Inicializar matriz resultado
    matC = np.zeros((m, p), dtype=np.float64)
    
    print("\nIniciando multiplicação distribuída...")
    start_time = time.perf_counter()
    
    def execute_task(uri, s_row, e_row, a_chunk, b_bytes, b_shape, dt_str, blk):
        # Criar novo proxy dentro da thread
        with Pyro5.api.Proxy(uri) as server:
            # Serializar apenas o pedaço de A necessário
            a_bytes = a_chunk.tobytes()
            a_shape = a_chunk.shape
            
            res_bytes = server.compute_partial_multiplication(
                a_bytes, b_bytes, a_shape, b_shape, dt_str, 
                s_row, e_row, blk
            )
        return s_row, e_row, res_bytes
    
    # Executar threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_task = {
            executor.submit(
                execute_task, 
                t['server_uri'], t['start_row'], t['end_row'], 
                matA[t['start_row']:t['end_row']], B_bytes, matB.shape, dtype_str, block_size
            ): t for t in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                s_row, e_row, result_bytes = future.result()
                
                # Inserir o pedaço calculado na matriz final
                if result_bytes:
                    # Decodificar se for dict (Serpent)
                    if isinstance(result_bytes, dict) and 'data' in result_bytes and 'encoding' in result_bytes:
                        if result_bytes['encoding'] == 'base64':
                            import base64
                            result_bytes = base64.b64decode(result_bytes['data'])

                    # Reconstrói o array numpy a partir dos bytes recebidos
                    # O shape é (linhas_processadas, colunas_B)
                    rows_processed = e_row - s_row
                    cols = p
                    result_array = np.frombuffer(result_bytes, dtype=np.dtype(dtype_str)).reshape(rows_processed, cols)
                    matC[s_row:e_row, :] = result_array
            except Exception as exc:
                print(f"  ✗ Exceção em uma tarefa: {exc}")
                raise exc
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print(f"✓ Multiplicação concluída em {elapsed_time:.4f}s")
    
    # Calcular total de cores
    total_cores = sum(info['num_cores'] for info in servers_info)
    
    return matC, elapsed_time, total_cores, valid_server_count