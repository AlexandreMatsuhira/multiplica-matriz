#!/usr/bin/env python3
"""
Script para executar multiplicação linear (single-core)
"""
import argparse
import sys
from pathlib import Path

from utils import load_matrix, save_matrix, run_benchmark, append_stats
from multiplication import multiply_linear
from config import DEFAULT_BLOCK_SIZE


def main():
    parser = argparse.ArgumentParser(
        description='Multiplicação linear de matrizes'
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
    print("MULTIPLICAÇÃO LINEAR DE MATRIZES")
    print("=" * 70)
    
    # Executar benchmark se solicitado
    benchmark_data = None
    if args.benchmark:
        benchmark_data = run_benchmark()
    
    # Carregar matrizes
    print(f"\nCarregando matriz A de: {args.matA}")
    try:
        matA = load_matrix(args.matA)
        print(f"  Dimensões: {matA.shape}")
    except Exception as e:
        print(f"Erro ao carregar matriz A: {e}")
        sys.exit(1)
    
    print(f"\nCarregando matriz B de: {args.matB}")
    try:
        matB = load_matrix(args.matB)
        print(f"  Dimensões: {matB.shape}")
    except Exception as e:
        print(f"Erro ao carregar matriz B: {e}")
        sys.exit(1)
    
    # Validar dimensões
    if matA.shape[1] != matB.shape[0]:
        print(f"\nERRO: Dimensões incompatíveis!")
        print(f"  Matriz A: {matA.shape}")
        print(f"  Matriz B: {matB.shape}")
        print(f"  O número de colunas de A ({matA.shape[1]}) deve ser igual")
        print(f"  ao número de linhas de B ({matB.shape[0]})")
        sys.exit(1)
    
    print(f"\nDimensões válidas para multiplicação:")
    print(f"  A: {matA.shape} x B: {matB.shape} = C: ({matA.shape[0]}, {matB.shape[1]})")
    
    # Executar multiplicação
    print(f"\nExecutando multiplicação linear...")
    print(f"  Núcleos: 1 (single-core)")
    print(f"  Block size: {args.block_size}")
    
    try:
        matC, elapsed_time = multiply_linear(matA, matB, args.block_size)
        
        print(f"\n✓ Multiplicação concluída!")
        print(f"  Tempo de execução: {elapsed_time:.6f} segundos")
        print(f"  Dimensões resultado: {matC.shape}")
        
    except Exception as e:
        print(f"\nErro durante multiplicação: {e}")
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
        "Linear": {
            "time_seconds": elapsed_time,
            "num_cores": 1,
            "block_size": args.block_size,
            "matrix_dimensions": {
                "A": list(matA.shape),
                "B": list(matB.shape),
                "C": list(matC.shape)
            }
        }
    }
    
    if benchmark_data:
        stats["Linear"]["benchmark"] = benchmark_data
    
    append_stats(stats)
    
    print("\n" + "=" * 70)
    print("EXECUÇÃO CONCLUÍDA COM SUCESSO")
    print("=" * 70)


if __name__ == "__main__":
    main()
