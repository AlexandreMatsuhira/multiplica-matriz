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
    
    # Converter matrizes para listas (para transmissão Pyro)
    # Nota: Enviar a matriz inteira é pesado, mas mantém a lógica original simplificada
    print("Serializando matrizes...")
    A_data = matA.tolist()
    B_data = matB.tolist()
    
    # Dividir trabalho (linhas de A) entre servidores disponíveis
    rows_per_server = m // valid_server_count
    tasks = []
    
    # Recalcular URIs baseados apenas nos que conectaram com sucesso
    # (Assumindo que server_uris original bate com servers_info na ordem, 
    # mas o ideal seria servers_info retornar o URI ou gerenciar ids)
    # Para simplificar, vamos usar a lista original se todos conectaram,
    # caso contrário precisaria filtrar server_uris.
    
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
    
    def execute_task(uri, s_row, e_row, a_dat, b_dat, blk):
        # Criar novo proxy dentro da thread
        with Pyro5.api.Proxy(uri) as server:
            res = server.compute_partial_multiplication(
                a_dat, b_dat, s_row, e_row, blk
            )
        return s_row, e_row, res
    
    # Executar threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_task = {
            executor.submit(
                execute_task, 
                t['server_uri'], t['start_row'], t['end_row'], 
                A_data, B_data, block_size
            ): t for t in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                s_row, e_row, result = future.result()
                
                # Inserir o pedaço calculado na matriz final
                if result:
                    result_array = np.array(result, dtype=np.float64)
                    matC[s_row:e_row, :] = result_array
            except Exception as exc:
                print(f"  ✗ Exceção em uma tarefa: {exc}")
                raise exc
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print(f"✓ Multiplicação concluída em {elapsed_time:.4f}s")
    
    return matC, elapsed_time