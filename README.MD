# Multiplicação de Matrizes - Análise de Desempenho

## Descrição

Projeto para análise de desempenho de multiplicação de matrizes usando diferentes abordagens:

- Linear (single-core)
- Paralela local (multi-core)
- Paralela distribuída (multi-core em 2 backends)

## Estrutura do Projeto

```
matrix_multiplication/
├── README.md
├── requirements.txt
├── config.py
├── utils/
│   ├── __init__.py
│   ├── matrix_io.py
│   ├── benchmark.py
│   └── stats.py
├── multiplication/
│   ├── __init__.py
│   ├── linear.py
│   ├── parallel_local.py
│   └── parallel_distributed.py
├── servers/
│   ├── __init__.py
│   └── computation_server.py
├── data/
│   ├── matA.txt
│   └── matB.txt
├── results/
│   └── stats.json
├── run_linear.py
├── run_parallel_local.py
├── run_parallel_distributed.py
├── start_server.py
└── analysis.ipynb
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

### 1. Executar Multiplicação Linear

```bash
python run_linear.py --matA data/matA.txt --matB data/matB.txt --output results/matC_linear.txt
```

### 2. Executar Multiplicação Paralela Local

```bash
python run_parallel_local.py --matA data/matA.txt --matB data/matB.txt --output results/matC_parallel.txt --cores 4
```

### 3. Executar Multiplicação Paralela Distribuída

**Terminal 1 - Servidor 1:**

```bash
python start_server.py --name server1 --port 9090
```

**Terminal 2 - Servidor 2:**

```bash
python start_server.py --name server2 --port 9091
```

**Terminal 3 - Cliente:**

```bash
python run_parallel_distributed.py --matA data/matA.txt --matB data/matB.txt --output results/matC_distributed.txt --servers localhost:9090,localhost:9091
```

### 4. Análise dos Resultados

```bash
jupyter notebook analysis.ipynb
```

## Formato dos Arquivos

### Matrizes de Entrada (matA.txt, matB.txt)

- Cada linha representa uma linha da matriz
- Valores separados por espaço único
- Valores entre 0.1500 e 1.1500 com 4 casas decimais
- Sem espaço após última coluna
- Sem linha em branco no final

Exemplo:

```
0.5000 0.7500 0.3200
0.1500 0.9800 0.4100
```

### Arquivo de Estatísticas (stats.json)

Gerado automaticamente após cada execução contendo:

- Tempo de execução (apenas multiplicação)
- Número de núcleos utilizados
- Tamanho do bloco (block_size)
- Benchmark da máquina

## Análises Geradas

1. **Gráfico de Tempo de Execução**: Comparação entre as três abordagens
2. **Benchmark**: Performance de inteiros e pontos flutuantes
3. **Speedup**: Análise e classificação (sublinear/linear/superlinear)
4. **Eficiência**: Cálculo da eficiência paralela
5. **Metodologia Foster**: Análise da aplicação distribuída

## Tecnologias Utilizadas

- **NumPy**: Estrutura de dados para matrizes
- **Numba**: JIT compilation para aceleração
- **Pyro5**: Comunicação distribuída
- **Multiprocessing**: Paralelização local com shared memory
- **Matplotlib/Seaborn**: Visualização de dados
- **Jupyter**: Análise interativa
