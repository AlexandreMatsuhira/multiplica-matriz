#!/usr/bin/env python3
"""
Script para gerar matrizes de teste
"""
import argparse
import numpy as np
from pathlib import Path


def generate_matrix(rows, cols, min_val=0.1500, max_val=1.1500, seed=None):
    """
    Gera uma matriz aleatória com valores no range especificado.
    
    Args:
        rows: Número de linhas
        cols: Número de colunas
        min_val: Valor mínimo
        max_val: Valor máximo
        seed: Seed para reprodutibilidade
        
    Returns:
        np.ndarray: Matriz gerada
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Gerar valores aleatórios
    matrix = np.random.uniform(min_val, max_val, size=(rows, cols))
    
    # Truncar em 4 casas decimais
    matrix = np.floor(matrix * 10000) / 10000
    
    return matrix


def save_matrix(matrix, filepath):
    """
    Salva matriz em arquivo texto com formato especificado.
    
    Args:
        matrix: Matriz a ser salva
        filepath: Caminho do arquivo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        rows, cols = matrix.shape
        for i in range(rows):
            row_values = []
            for j in range(cols):
                value = matrix[i, j]
                row_values.append(f"{value:.4f}")
            
            f.write(" ".join(row_values))
            
            # Não adicionar nova linha no último elemento
            if i < rows - 1:
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Gerar matrizes de teste para multiplicação'
    )
    parser.add_argument(
        '--rows-A',
        type=int,
        required=True,
        help='Número de linhas da matriz A'
    )
    parser.add_argument(
        '--cols-A',
        type=int,
        required=True,
        help='Número de colunas da matriz A'
    )
    parser.add_argument(
        '--cols-B',
        type=int,
        required=True,
        help='Número de colunas da matriz B (linhas = cols-A)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Diretório de saída (padrão: data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed para geração aleatória (padrão: 42)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GERADOR DE MATRIZES DE TESTE")
    print("=" * 70)
    
    # Gerar matriz A
    print(f"\nGerando matriz A ({args.rows_A} x {args.cols_A})...")
    matA = generate_matrix(args.rows_A, args.cols_A, seed=args.seed)
    
    output_A = Path(args.output_dir) / "matA.txt"
    save_matrix(matA, output_A)
    print(f"  ✓ Salva em: {output_A}")
    print(f"  Amostra (primeira linha): {matA[0, :min(5, args.cols_A)]}")
    
    # Gerar matriz B (compatível com A)
    print(f"\nGerando matriz B ({args.cols_A} x {args.cols_B})...")
    matB = generate_matrix(args.cols_A, args.cols_B, seed=args.seed + 1)
    
    output_B = Path(args.output_dir) / "matB.txt"
    save_matrix(matB, output_B)
    print(f"  ✓ Salva em: {output_B}")
    print(f"  Amostra (primeira linha): {matB[0, :min(5, args.cols_B)]}")
    
    # Informações sobre a multiplicação
    print(f"\nInformações da multiplicação:")
    print(f"  A: {matA.shape} x B: {matB.shape} = C: ({matA.shape[0]}, {matB.shape[1]})")
    print(f"  Total de operações: {matA.shape[0] * matB.shape[1] * matA.shape[1]:,}")
    
    # Estimativa de tamanho
    size_A = matA.nbytes / (1024 ** 2)
    size_B = matB.nbytes / (1024 ** 2)
    size_C = (matA.shape[0] * matB.shape[1] * 8) / (1024 ** 2)
    
    print(f"\nTamanhos em memória:")
    print(f"  Matriz A: {size_A:.2f} MB")
    print(f"  Matriz B: {size_B:.2f} MB")
    print(f"  Matriz C (resultado): {size_C:.2f} MB")
    print(f"  Total: {size_A + size_B + size_C:.2f} MB")
    
    print("\n" + "=" * 70)
    print("MATRIZES GERADAS COM SUCESSO")
    print("=" * 70)
    
    print("\nPróximos passos:")
    print("  1. Execute: python run_linear.py --matA data/matA.txt --matB data/matB.txt --output results/matC_linear.txt")
    print("  2. Execute: python run_parallel_local.py --matA data/matA.txt --matB data/matB.txt --output results/matC_parallel.txt")
    print("  3. Inicie servidores e execute: python run_parallel_distributed.py ...")
    print("  4. Analise: jupyter notebook analysis.ipynb")


if __name__ == "__main__":
    main()