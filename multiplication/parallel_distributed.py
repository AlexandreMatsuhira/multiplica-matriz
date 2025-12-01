"""
Multiplicação paralela distribuída usando Pyro5
"""
import numpy as np
import time
import Pyro5.api


def multiply_parallel_distributed(matA, matB, server_uris, block_size=64):
    """
    Multiplica duas matrizes usando servidores distribuídos.
    
    Args:
        matA: Matriz A (NumPy array)
        matB: Matriz B (NumPy array)
        server_uris: Lista de URIs dos servidores Pyro
        block_size: Tamanho do bloco para tiling
        
    Returns:
        tuple: (matC, tempo_execucao)
    """
    # Validar dimensões
    if matA.shape[1] != matB.shape[0]:
        raise ValueError(
            f"Dimensões incompatíveis: {matA.shape} x {matB.shape}"
        )
    
    m, n = matA.shape
    p = matB.shape[1]
    
    # Conectar aos servidores
    servers = []
    total_cores = 0
    
    print(f"\nConectando a {len(server_uris)} servidores...")
    for uri in server_uris:
        try:
            server = Pyro5.api.Proxy(uri)
            info = server.get_info()
            servers.append(info)
            total_cores += info['num_cores']
            print(f"  ✓ {info['name']}: {info['num_cores']} núcleos")
            server._pyroRelease()  # Liberar proxy após uso
        except Exception as e:
            print(f"  ✗ Erro ao conectar a {uri}: {e}")
    
    if not servers:
        raise RuntimeError("Nenhum servidor disponível")
    
    print(f"\nTotal de núcleos disponíveis: {total_cores}")
    
    # Converter matrizes para listas (para transmissão Pyro)
    A_data = matA.tolist()
    B_data = matB.tolist()
    
    # Dividir trabalho entre servidores
    rows_per_server = m // len(server_uris)
    tasks = []
    
    for i in range(len(server_uris)):
        start_row = i * rows_per_server
        if i == len(server_uris) - 1:
            end_row = m  # Último servidor pega linhas restantes
        else:
            end_row = (i + 1) * rows_per_server
        
        tasks.append({
            'server_uri': server_uris[i],
            'start_row': start_row,
            'end_row': end_row,
            'rows': end_row - start_row
        })
    
    print(f"\nDistribuindo {m} linhas entre {len(server_uris)} servidores...")
    for i, task in enumerate(tasks):
        print(f"  Servidor {i+1}: linhas {task['start_row']}-{task['end_row']} "
              f"({task['rows']} linhas)")
    
    # Inicializar matriz resultado
    matC = np.zeros((m, p), dtype=np.float64)
    
    # Medir apenas o tempo de multiplicação
    print("\nIniciando multiplicação distribuída...")
    start_time = time.perf_counter()
    
    # Executar tarefas em paralelo usando threads
    import concurrent.futures
    
    def execute_task(server_uri, start_row, end_row, A_data, B_data, block_size):
        """
        Executa uma tarefa em um servidor.
        Cria um novo proxy dentro da thread para evitar problemas de ownership.
        """
        # Criar novo proxy dentro desta thread
        server = Pyro5.api.Proxy(server_uri)
        
        result = server.compute_partial_multiplication(
            A_data, B_data, start_row, end_row, block_size
        )
        
        server._pyroRelease()
        
        return start_row, end_row, result
    
    # Preparar tarefas 
    task_args = []
    for i, task in enumerate(tasks):
        task_args.append((
            server_uris[i],
            task['start_row'],
            task['end_row'],
            A_data,
            B_data,
            block_size
        ))
    
    # Executar todas as tarefas em paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(execute_task, *args) for args in task_args]
        
        # Coletar resultados
        for future in concurrent.futures.as_completed(futures):
            start_row, end_row, result = future.result()
            result_array = np.array(result, dtype=np.float64)
            matC[start_row:end_row, :] = result_array
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    print(f"✓ Multiplicação concluída em {elapsed_time:.4f}s")
    
    return matC, elapsed_time