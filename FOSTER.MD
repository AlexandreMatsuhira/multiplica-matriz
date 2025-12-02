### Análise da Aplicação Distribuída (Metodologia Foster)

**1. Decomposição (Partitioning):**
O problema da multiplicação de matrizes $C = A \times B$ foi dividido em tarefas menores (blocos de $64 \times 64$, conforme indicado nos metadados). Isso permitiu granularidade fina.

**2. Comunicação (Communication):**
Houve necessidade de troca intensa de dados entre os nós para compartilhar as linhas de A e colunas de B. O tempo elevado no modo distribuído indica que o custo dessa comunicação (latência de rede) superou o ganho computacional.

**3. Aglomeração (Agglomeration):**
As tarefas foram agrupadas para serem enviadas aos 2 servidores disponíveis. No entanto, para uma matriz pequena (500x500), a proporção de *cálculo vs. comunicação* foi desfavorável.

**4. Mapeamento (Mapping):**
As tarefas foram distribuídas estaticamente ou dinamicamente entre os 2 *workers*. O balanceamento de carga ocorreu, mas o gargalo de I/O de rede impediu o Speedup positivo.