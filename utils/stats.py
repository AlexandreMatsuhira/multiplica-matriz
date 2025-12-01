"""
Funções para gerenciamento de estatísticas
"""
import json
from pathlib import Path
from config import STATS_FILE


def save_stats(stats, filepath=None):
    """
    Salva estatísticas em arquivo JSON.
    
    Args:
        stats: Lista de dicionários com estatísticas
        filepath: Caminho do arquivo (usa STATS_FILE se None)
    """
    if filepath is None:
        filepath = STATS_FILE
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nEstatísticas salvas em: {filepath}")


def load_stats(filepath=None):
    """
    Carrega estatísticas de arquivo JSON.
    
    Args:
        filepath: Caminho do arquivo (usa STATS_FILE se None)
        
    Returns:
        list: Lista de estatísticas ou lista vazia se arquivo não existir
    """
    if filepath is None:
        filepath = STATS_FILE
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        return []
    
    with open(filepath, 'r') as f:
        return json.load(f)


def append_stats(new_stat, filepath=None):
    """
    Adiciona nova estatística ao arquivo existente.
    
    Args:
        new_stat: Dicionário com nova estatística
        filepath: Caminho do arquivo (usa STATS_FILE se None)
    """
    stats = load_stats(filepath)
    stats.append(new_stat)
    save_stats(stats, filepath)


def calculate_speedup(time_sequential, time_parallel):
    """
    Calcula o speedup.
    
    Args:
        time_sequential: Tempo de execução sequencial
        time_parallel: Tempo de execução paralelo
        
    Returns:
        float: Speedup
    """
    if time_parallel == 0:
        return float('inf')
    return time_sequential / time_parallel


def classify_speedup(speedup, num_processors):
    """
    Classifica o speedup.
    
    Args:
        speedup: Valor do speedup
        num_processors: Número de processadores usados
        
    Returns:
        str: Classificação (sublinear, linear, superlinear)
    """
    if speedup < num_processors * 0.9:
        return "sublinear"
    elif speedup <= num_processors * 1.1:
        return "linear"
    else:
        return "superlinear"


def calculate_efficiency(speedup, num_processors):
    """
    Calcula a eficiência paralela.
    
    Args:
        speedup: Valor do speedup
        num_processors: Número de processadores usados
        
    Returns:
        float: Eficiência (0-1)
    """
    if num_processors == 0:
        return 0.0
    return speedup / num_processors


def format_stats_for_analysis(stats_list):
    """
    Formata estatísticas para análise.
    
    Args:
        stats_list: Lista de estatísticas
        
    Returns:
        dict: Estatísticas formatadas
    """
    analysis = {}
    
    for stat_dict in stats_list:
        for mode, data in stat_dict.items():
            analysis[mode] = {
                "time": data.get("time_seconds", 0),
                "cores": data.get("num_cores", 1),
                "block_size": data.get("block_size", 0)
            }
    
    # Calcular speedup e eficiência se houver dados Linear
    if "Linear" in analysis:
        base_time = analysis["Linear"]["time"]
        
        for mode in analysis:
            if mode != "Linear":
                cores = analysis[mode]["cores"]
                speedup = calculate_speedup(base_time, analysis[mode]["time"])
                efficiency = calculate_efficiency(speedup, cores)
                classification = classify_speedup(speedup, cores)
                
                analysis[mode]["speedup"] = speedup
                analysis[mode]["efficiency"] = efficiency
                analysis[mode]["classification"] = classification
    
    return analysis
