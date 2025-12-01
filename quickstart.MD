# Guia Rápido de Uso

## Instalação Rápida

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# Ou usando Make
make install
```

## Teste Rápido (Matrizes Pequenas)

```bash
# Usando Make (recomendado)
make test-small

# Ou manualmente:
python generate_test_matrices.py --rows-A 100 --cols-A 100 --cols-B 100
python run_linear.py --matA data/matA.txt --matB data/matB.txt --output results/matC_linear.txt --benchmark
python run_parallel_local.py --matA data/matA.txt --matB data/matB.txt --output results/matC_parallel.txt --benchmark
jupyter notebook analysis.ipynb
```

## Uso Detalhado

### 1. Gerar Matrizes de Teste

```bash
# Matrizes pequenas (100x100) - para testes rápidos
make generate-small

# Matrizes médias (500x500) - para análise equilibrada
make generate-medium

# Matrizes grandes (1000x1000) - para teste de performance
make generate-large

# Personalizado
python generate_test_matrices.py --rows-A 800 --cols-A 600 --cols-B 400
```

### 2. Executar Multiplicação Linear

```bash
python run_linear.py \
    --matA data/matA.txt \
    --matB data/matB.txt \
    --output results/matC_linear.txt \
    --benchmark

# Ou usando Make
make run-linear
```

**Saída esperada:**
- Arquivo `results/matC_linear.txt` com resultado
- Entrada adicionada em `results/stats.json`
- Tempo de execução exibido no console

### 3. Executar Multiplicação Paralela Local

```bash
python run_parallel_local.py \
    --matA data/matA.txt \
    --matB data/matB.txt \
    --output results/matC_parallel.txt \
    --cores 4 \
    --benchmark

# Ou usando Make (usa todos os núcleos disponíveis)
make run-parallel
```

**Opções:**
- `--cores N`: Número de núcleos (padrão: todos disponíveis)
- `--block-size N`: Tamanho do bloco para tiling (padrão: 64)

### 4. Executar Multiplicação Paralela Distribuída

**Passo 1: Iniciar Servidores**

Terminal 1:
```bash
python start_server.py --name server1 --port 9090
```

Terminal 2:
```bash
python start_server.py --name server2 --port 9091
```

**Passo 2: Executar Cliente**

Terminal 3:
```bash
python run_parallel_distributed.py \
    --matA data/matA.txt \
    --matB data/matB.txt \
    --output results/matC_distributed.txt \
    --servers PYRO:server1@localhost:9090,PYRO:server2@localhost:9091 \
    --benchmark
```

**Formato simplificado para servidores locais:**
```bash
python run_parallel_distributed.py \
    --matA data/matA.txt \
    --matB data/matB.txt \
    --output results/matC_distributed.txt \
    --servers localhost:9090,localhost:9091
```

### 5. Analisar Resultados

```bash
# Abrir notebook de análise
jupyter notebook analysis.ipynb

# Ou usando Make
make analysis
```

**O notebook fornece:**
- Gráfico comparativo de tempo de execução
- Análise de speedup e eficiência
- Classificação do speedup (sublinear/linear/superlinear)
- Benchmark do hardware
- Análise segundo Metodologia Foster

## Fluxo de Trabalho Completo

```bash
# 1. Limpar resultados anteriores
make clean

# 2. Gerar matrizes de teste
make generate-medium

# 3. Executar todos os modos (exceto distribuído)
make run-all

# 4. (Opcional) Executar modo distribuído
# Terminal 1:
python start_server.py --name server1 --port 9090
# Terminal 2:
python start_server.py --name server2 --port 9091
# Terminal 3:
python run_parallel_distributed.py --matA data/matA.txt --matB data/matB.txt \
    --output results/matC_distributed.txt --servers localhost:9090,localhost:9091

# 5. Analisar resultados
make analysis
```

## Estrutura do stats.json

Após cada execução, o arquivo `results/stats.json` é atualizado:

```json
[
    {
        "Linear": {
            "time_seconds": 0.9362630540272221,
            "num_cores": 1,
            "block_size": 64,
            "matrix_dimensions": {
                "A": [500, 500],
                "B": [500, 500],
                "C": [500, 500]
            },
            "benchmark": {
                "system_info": {...},
                "integer_benchmark": {...},
                "float_benchmark": {...}
            }
        }
    },
    {
        "Parallel_Local": {
            "time_seconds": 0.2451234567890123,
            "num_cores": 8,
            "block_size": 64,
            "matrix_dimensions": {...}
        }
    },
    {
        "Parallel_Distributed": {
            "time_seconds": 0.1823456789012345,
            "num_servers": 2,
            "num_cores": "distributed",
            "block_size": 64,
            "matrix_dimensions": {...}
        }
    }
]
```

## Dicas de Performance

### Para Matrizes Pequenas (< 500x500)
- Linear pode ser mais rápido devido ao overhead de paralelização
- Use `--block-size 32`

### Para Matrizes Médias (500-2000)
- Paralelização local mostra bons resultados
- Use `--block-size 64` ou `--block-size 128`
- Configure número de cores apropriado

### Para Matrizes Grandes (> 2000)
- Paralelização distribuída traz benefícios significativos
- Use `--block-size 128` ou maior
- Considere usar mais servidores

### Otimização de Block Size
```bash
# Testar diferentes tamanhos
for bs in 32 64 128 256; do
    python run_parallel_local.py \
        --matA data/matA.txt \
        --matB data/matB.txt \
        --output results/matC_bs${bs}.txt \
        --block-size $bs
done
```

## Troubleshooting

### Erro: "Dimensões incompatíveis"
- Verifique que cols_A = rows_B
- Regenere as matrizes corretamente

### Erro de memória
- Reduza o tamanho das matrizes
- Use modo distribuído para matrizes muito grandes

### Servidor Pyro não conecta
- Verifique se o servidor está rodando
- Confirme porta e host corretos
- Verifique firewall

### Performance ruim no modo paralelo
- Ajuste `--block-size`
- Verifique se há outros processos consumindo CPU
- Para matrizes pequenas, linear pode ser mais rápido

## Exemplos de Comandos Prontos

```bash
# Teste completo rápido (matrizes 100x100)
make test-small

# Teste de performance (matrizes 1000x1000)
make test-large

# Comparar diferentes números de cores
for cores in 1 2 4 8; do
    python run_parallel_local.py --matA data/matA.txt --matB data/matB.txt \
        --output results/matC_${cores}cores.txt --cores $cores
done

# Executar apenas análise (se já tiver dados)
make analysis
```
