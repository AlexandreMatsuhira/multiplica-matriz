"""
Funções para leitura e escrita de matrizes
"""
import numpy as np
from pathlib import Path


def load_matrix(filepath):
    """
    Carrega uma matriz de um arquivo texto.
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        np.ndarray: Matriz carregada
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Ignora linhas vazias
                row = [float(x) for x in line.split()]
                matrix.append(row)
    
    return np.array(matrix, dtype=np.float64)


def save_matrix(matrix, filepath):
    """
    Salva uma matriz em um arquivo texto com truncamento (não arredondamento).
    
    Args:
        matrix: np.ndarray - Matriz a ser salva
        filepath: Caminho do arquivo de saída
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        rows, cols = matrix.shape
        for i in range(rows):
            row_values = []
            for j in range(cols):
                # Truncar em 4 casas decimais (não arredondar)
                value = matrix[i, j]
                truncated = np.floor(value * 10000) / 10000
                row_values.append(f"{truncated:.4f}")
            
            # Escrever linha sem espaço no final
            f.write(" ".join(row_values))
            
            # Não adicionar nova linha no último elemento
            if i < rows - 1:
                f.write("\n")


def validate_matrix_dimensions(matA, matB):
    """
    Valida se as dimensões das matrizes são compatíveis para multiplicação.
    
    Args:
        matA: Primeira matriz
        matB: Segunda matriz
        
    Returns:
        bool: True se compatíveis
        
    Raises:
        ValueError: Se dimensões incompatíveis
    """
    if matA.shape[1] != matB.shape[0]:
        raise ValueError(
            f"Dimensões incompatíveis: matA {matA.shape} x matB {matB.shape}. "
            f"O número de colunas de A ({matA.shape[1]}) deve ser igual "
            f"ao número de linhas de B ({matB.shape[0]})."
        )
    return True
