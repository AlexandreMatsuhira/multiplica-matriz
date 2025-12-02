#!/usr/bin/env python3
"""
Script para executar multiplicação paralela distribuída
"""
import argparse
import sys

from utils import load_matrix, save_matrix, run_benchmark, append_stats
from multiplication import multiply_parallel_distributed
from config import DEFAULT_BLOCK_SIZE


def main():
    parser = argparse.ArgumentParser(
        description='Multiplicação paralela distribuída de matrizes'
    )
    parser.add_argument(
        '--matA',
        type=str,
        required=True,
        help='Caminho do arquivo da matriz A'
    )
    parser.add_argument(
        '--matB',
        type=str,
        required=True,
        help='Caminho do arquivo da matriz B'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Caminho do arquivo de saída para matriz C'
    )
    parser.add_argument(
        '--servers',
        type=str,
        required=True,
        help='URIs dos servidores separados por vírgula (ex: PYRO:server1@localhost:9090,PYRO:server2@localhost:9091)'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f'Tamanho do bloco para tiling (padrão: {DEFAULT_BLOCK_SIZE})'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Executar benchmark do sistema'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MULTIPLICAÇÃO PARALELA DISTRIBUÍDA DE MATRIZES")
    print("=" * 70)
    
    # Parse server URIs
    print("\n1. Parsing configuração de servidores...")
    server_uris = []
    for i, server_str in enumerate(args.servers.split(',')):
        server_str = server_str.strip()
        
        if not server_str:
            continue
            
        if server_str.startswith('PYRO:'):
            # URI completa fornecida
            uri = server_str
        elif ':' in server_str:
            # Formato: host:port -> PYRO:serverN@host:port
            parts = server_str.split(':')
            if len(parts) == 2:
                host, port = parts
                server_name = f"server{i+1}"
                uri = f"PYRO:{server_name}@{host}:{port}"
            else:
                print(f"  ✗ Formato inválido para servidor: {server_str}")
                print(f"    Use: host:port ou PYRO:name@host:port")
                continue
        else:
            # Apenas porta, assume localhost
            try:
                port = int(server_str)
                server_name = f"server{i+1}"
                uri = f"PYRO:{server_name}@localhost:{port}"
            except ValueError:
                print(f"  ✗ Formato inválido para servidor: {server_str}")
                print(f"    Use: port, host:port ou PYRO:name@host:port")
                continue
        
        server_uris.append(uri)
        print(f"  ✓ Servidor {i+1}: {uri}")
    
    if not server_uris:
        print("\n✗ Erro: Nenhum servidor válido especificado")
        print("\nExemplos de uso:")
        print("  --servers 9090")
        print("  --servers localhost:9090")
        print("  --servers localhost:9090,localhost:9091")
        print("  --servers PYRO:server1@localhost:9090")
        sys.exit(1)
    
    # Executar benchmark se solicitado
    benchmark_data = None
    if args.benchmark:
        benchmark_data = run_benchmark()
    
    # Carregar matrizes
    print(f"\n2. Carregando matrizes...")
    print(f"  Matriz A: {args.matA}")
    try:
        matA = load_matrix(args.matA)
        print(f"    ✓ Dimensões: {matA.shape}")
    except Exception as e:
        print(f"    ✗ Erro ao carregar matriz A: {e}")
        sys.exit(1)
    
    print(f"  Matriz B: {args.matB}")
    try:
        matB = load_matrix(args.matB)
        print(f"    ✓ Dimensões: {matB.shape}")
    except Exception as e:
        print(f"    ✗ Erro ao carregar matriz B: {e}")
        sys.exit(1)
    
    # Validar dimensões
    if matA.shape[1] != matB.shape[0]:
        print(f"\n✗ ERRO: Dimensões incompatíveis!")
        print(f"  Matriz A: {matA.shape}")
        print(f"  Matriz B: {matB.shape}")
        print(f"  O número de colunas de A ({matA.shape[1]}) deve ser igual")
        print(f"  ao número de linhas de B ({matB.shape[0]})")
        sys.exit(1)
    
    print(f"\n3. Validação:")
    print(f"  ✓ Dimensões compatíveis: A{matA.shape} × B{matB.shape} = C({matA.shape[0]}, {matB.shape[1]})")
    
    # Executar multiplicação
    print(f"\n4. Executando multiplicação distribuída...")
    print(f"  Servidores: {len(server_uris)}")
    print(f"  Block size: {args.block_size}")
    
    try:
        matC, elapsed_time, total_cores, num_servers = multiply_parallel_distributed(
            matA, matB, server_uris, args.block_size
        )
        
        print(f"\n✓ Multiplicação distribuída concluída!")
        print(f"  Tempo de execução: {elapsed_time:.6f} segundos")
        print(f"  Dimensões resultado: {matC.shape}")
        
    except Exception as e:
        print(f"\nErro durante multiplicação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Salvar resultado
    print(f"\nSalvando resultado em: {args.output}")
    try:
        save_matrix(matC, args.output)
        print("  ✓ Matriz salva com sucesso")
    except Exception as e:
        print(f"Erro ao salvar matriz: {e}")
        sys.exit(1)
    
    # Salvar estatísticas
    stats = {
        "Parallel_Distributed": {
            "time_seconds": elapsed_time,
            "num_servers": num_servers,
            "num_cores": total_cores, 
            "block_size": args.block_size,
            "matrix_dimensions": {
                "A": list(matA.shape),
                "B": list(matB.shape),
                "C": list(matC.shape)
            }
        }
    }
    
    if benchmark_data:
        stats["Parallel_Distributed"]["benchmark"] = benchmark_data
    
    append_stats(stats)
    
    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO")
    print("=" * 70)


if __name__ == "__main__":
    main()