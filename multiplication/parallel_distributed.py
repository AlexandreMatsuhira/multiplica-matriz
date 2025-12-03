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
    
    total_linhas_A, total_colunas_A = matA.shape
    total_colunas_B = matB.shape[1]
    
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
    linhas_por_servidor = total_linhas_A // valid_server_count
    tasks = []
    
    # Recalcular URIs baseados apenas nos que conectaram com sucesso
    active_uris = server_uris[:valid_server_count] 

    for indice_servidor in range(valid_server_count):
        linha_inicial = indice_servidor * linhas_por_servidor
        if indice_servidor == valid_server_count - 1:
            linha_final = total_linhas_A  # Último servidor pega o resto
        else:
            linha_final = (indice_servidor + 1) * linhas_por_servidor
        
        tasks.append({
            'server_uri': active_uris[indice_servidor],
            'start_row': linha_inicial,
            'end_row': linha_final,
            'rows': linha_final - linha_inicial
        })
    
    print(f"\nDistribuindo {total_linhas_A} linhas entre {valid_server_count} servidores...")
    for i, task in enumerate(tasks):
        print(f"  Servidor {i+1}: linhas {task['start_row']}-{task['end_row']} "
              f"({task['rows']} linhas)")
    
    # Inicializar matriz resultado
    matC = np.zeros((total_linhas_A, total_colunas_B), dtype=np.float64)
    
    print("\nIniciando multiplicação distribuída...")
    start_time = time.perf_counter()
    
    def execute_task(uri, linha_ini, linha_fim, a_chunk, b_bytes, b_shape, dt_str, blk):
        # Criar novo proxy dentro da thread
        with Pyro5.api.Proxy(uri) as server:
            # Serializar apenas o pedaço de A necessário
            a_bytes = a_chunk.tobytes()
            a_shape = a_chunk.shape
            
            res_bytes = server.compute_partial_multiplication(
                a_bytes, b_bytes, a_shape, b_shape, dt_str, 
                linha_ini, linha_fim, blk
            )
        return linha_ini, linha_fim, res_bytes
    
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
                linha_ini, linha_fim, result_bytes = future.result()
                
                # Inserir o pedaço calculado na matriz final
                if result_bytes:
                    # Decodificar se for dict (Serpent)
                    if isinstance(result_bytes, dict) and 'data' in result_bytes and 'encoding' in result_bytes:
                        if result_bytes['encoding'] == 'base64':
                            import base64
                            result_bytes = base64.b64decode(result_bytes['data'])

                    # Reconstrói o array numpy a partir dos bytes recebidos
                    # O shape é (linhas_processadas, colunas_B)
                    rows_processed = linha_fim - linha_ini
                    cols = total_colunas_B
                    result_array = np.frombuffer(result_bytes, dtype=np.dtype(dtype_str)).reshape(rows_processed, cols)
                    matC[linha_ini:linha_fim, :] = result_array
            except Exception as exc:
                print(f"  ✗ Exceção em uma tarefa: {exc}")
                raise exc
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print(f"✓ Multiplicação concluída em {elapsed_time:.4f}s")
    
    # Calcular total de cores
    total_cores = sum(info['num_cores'] for info in servers_info)
    
    return matC, elapsed_time, total_cores, valid_server_count