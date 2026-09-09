# Tradução Automática de Baixo Recurso — Português ↔ Tupi Antigo

Este projeto investiga a viabilidade de tradução automática entre **Português (por_Latn)** e **Tupi Antigo**, representado pelo código **grn_Latn**, em um cenário de **baixa disponibilidade de dados paralelos**.

Foram avaliados dois regimes:

* **Zero-shot:** utilização direta de um modelo multilíngue pré-treinado, sem ajuste específico para o par Português–Tupi.
* **Few-shot:** fine-tuning supervisionado do modelo utilizando **LoRA (PEFT)**, com modelos independentes para cada direção de tradução.

A avaliação combina métricas automáticas (**BLEU, chrF1 e chrF3**) com uma análise qualitativa das melhores e piores traduções segundo o BLEU por sentença.

## Objetivos

O projeto busca:

1. Avaliar a capacidade **zero-shot** do modelo NLLB para o par Português–Tupi.
2. Adaptar o modelo ao domínio por meio de **fine-tuning com LoRA**.
3. Comparar quantitativamente os regimes zero-shot e few-shot.
4. Analisar qualitativamente os erros e acertos produzidos pelo modelo.
5. Investigar a assimetria entre as direções **Português → Tupi** e **Tupi → Português**.

## Modelo

O modelo utilizado foi:

```text
facebook/nllb-200-distilled-600M
```

A escolha foi motivada principalmente pelo suporte nativo ao código `grn_Latn`, pela menor demanda computacional em relação às versões maiores e pela compatibilidade com métodos de adaptação eficiente como **PEFT/LoRA**. O treinamento foi realizado integralmente em **CPU**, devido a problemas de compatibilidade entre a GPU disponível e o PyTorch.

## Instalação

### 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

### 2. Criar um ambiente virtual

Recomenda-se utilizar Python 3.10 ou superior.

Com `venv`:

```bash
python -m venv .venv
```

Ativação no Linux/macOS:

```bash
source .venv/bin/activate
```

Ativação no Windows:

```powershell
.venv\Scripts\activate
```

### 3. Instalar as dependências

Instale as bibliotecas utilizadas pelo notebook:

```bash
pip install torch
pip install transformers
pip install datasets
pip install peft
pip install accelerate
pip install sacrebleu
pip install pandas
pip install openpyxl
pip install jupyter
```

Também é possível instalar todas de uma vez:

```bash
pip install torch transformers datasets peft accelerate sacrebleu pandas openpyxl jupyter
```

> **Observação:** o experimento foi executado em CPU. O treinamento do modelo `nllb-200-distilled-600M` é computacionalmente custoso, especialmente durante o fine-tuning.

## Execução

O pipeline completo está implementado no notebook:

```text
ep2.ipynb
```

Para iniciar o Jupyter Notebook:

```bash
jupyter notebook
```

ou:

```bash
jupyter lab
```

Abra então:

```text
ep2.ipynb
```

e execute as células em ordem.

O notebook contém as etapas de:

```text
Pré-processamento
      ↓
Divisão train / validation / test
      ↓
Inferência zero-shot
      ↓
Fine-tuning com LoRA
      ↓
Seleção do melhor checkpoint
      ↓
Tradução do conjunto de teste
      ↓
BLEU / chrF1 / chrF3
      ↓
Seleção das melhores e piores traduções
```

O relatório descreve esse pipeline como: limpeza e normalização do corpus, divisão 70/15/15, inferência zero-shot, fine-tuning com LoRA em cada direção, geração das predições no teste e avaliação automática.

### Execução somente da avaliação

As traduções já geradas podem ser utilizadas para reproduzir a análise sem executar novamente todo o processo de inferência ou treinamento.

Os principais arquivos de resultados são:

```text
results_zero_shot.txt
results_few_shot.txt
```

As traduções processadas encontram-se em:

```text
pt2tupi_zeroShot_fixed.csv
tupi2pt_zeroShot_fixed.csv
pt2tupi_fewShot_fixed.csv
tupi2pt_fewShot_fixed.csv
```

As listas das dez melhores e dez piores traduções estão disponíveis nos respectivos arquivos `best10` e `worst10`.

## Corpus

O corpus paralelo contém as colunas:

```text
portugues
tupi
```

O corpus original está disponível em:

https://github.com/CalebeRezende/oldtupi_dataset/blob/main/C%C3%B3pia%20de%20portugues-guarani-tupi%20antigo.xlsx

### Pré-processamento

Foram realizadas as seguintes etapas:

* remoção de conteúdo entre parênteses;
* normalização dos apóstrofos;
* remoção de aspas exteriores inconsistentes;
* padronização das linhas CSV;
* substituição de `ñ` por `nh` no Tupi;
* normalização Unicode em NFC;
* remoção de espaços duplicados;
* embaralhamento do corpus;
* divisão em **70% treino, 15% validação e 15% teste**.

## Pipeline

```text
Corpus original
      │
      ▼
Limpeza e normalização
      │
      ▼
Train / Validation / Test
      │
      ├──────────────► Zero-shot
      │                   │
      │                   ▼
      │              NLLB pré-treinado
      │
      └──────────────► Few-shot
                          │
                          ▼
                     LoRA / PEFT
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
         PT → Tupi                Tupi → PT
              │                       │
              └───────────┬───────────┘
                          ▼
                   Tradução do teste
                          │
                          ▼
             BLEU / chrF1 / chrF3
                          │
                          ▼
                Top 10 / Bottom 10
```

## Zero-shot

No regime zero-shot, o `facebook/nllb-200-distilled-600M` é utilizado diretamente em seu estado pré-treinado.

O tokenizer nativo do NLLB é configurado com `src_lang` e `tgt_lang` adequados a cada direção. As traduções são geradas com limite de **128 tokens** e armazenadas em CSV para evitar recomputações, devido ao alto custo computacional da inferência em CPU.

## Few-shot

No regime few-shot, foram treinados **dois modelos independentes**:

```text
PT → Tupi
Tupi → PT
```

Para cada direção foi utilizado um tokenizer específico e um adaptador **LoRA** com:

```text
r = 16
alpha = 32
target_modules = q_proj, v_proj
```

O treinamento foi configurado com:

```text
epochs        = 5
learning rate = 5e-5
weight decay  = 0.01
batch size    = 4
early stopping patience = 2
```

A seleção final foi feita utilizando o **maior BLEU na validação**.

### Melhor checkpoint

| Direção   | Melhor época |     BLEU |
| --------- | -----------: | -------: |
| PT → Tupi |            2 | 0.470439 |
| Tupi → PT |            2 | 0.988228 |

## Resultados

| Direção   | Regime    |       BLEU |       chrF1 |       chrF3 |
| --------- | --------- | ---------: | ----------: | ----------: |
| PT → Tupi | Zero-shot |     0.1190 |     11.9806 |     13.2296 |
| PT → Tupi | Few-shot  | **0.4611** |      6.8467 |      9.1878 |
| Tupi → PT | Zero-shot |     0.2404 |     12.3135 |     12.0392 |
| Tupi → PT | Few-shot  | **0.8308** | **13.9667** | **13.8709** |

O fine-tuning aumentou substancialmente o BLEU nas duas direções, embora os resultados das métricas baseadas em caracteres tenham apresentado comportamento assimétrico.

## Análise qualitativa

O modelo zero-shot apresenta algumas traduções plausíveis, principalmente em expressões curtas e frequentes, mas também produz traduções semanticamente desvinculadas do texto original.

No regime few-shot, o modelo apresenta maior sensibilidade ao vocabulário e às estruturas do corpus, mas também surgem comportamentos degenerados, principalmente **repetições patológicas de morfemas e fragmentos fonológicos**.

## Estrutura dos arquivos

```text
.
├── ep2.ipynb
│
├── train_fixed.csv
├── validation_fixed.csv
├── test_fixed.csv
│
├── pt2tupi_zeroShot_fixed.csv
├── tupi2pt_zeroShot_fixed.csv
├── pt2tupi_fewShot_fixed.csv
├── tupi2pt_fewShot_fixed.csv
│
├── pt2tupi_zeroShot_best10.csv
├── pt2tupi_zeroShot_worst10.csv
├── tupi2pt_zeroShot_best10.csv
├── tupi2pt_zeroShot_worst10.csv
│
├── pt2tupi_fewShot_best10.csv
├── pt2tupi_fewShot_worst10.csv
├── tupi2pt_fewShot_best10.csv
├── tupi2pt_fewShot_worst10.csv
│
├── results_zero_shot.txt
├── results_few_shot.txt
│
├── Cópia de portugues-guarani-tupi antigo.xlsx
│
├── nllb-ft-pt2tupi/
└── nllb-ft-tupi2pt/
```

Os arquivos acima correspondem aos artefatos descritos no relatório.

## Limitações

Os resultados devem ser interpretados considerando:

* o tamanho reduzido do corpus;
* a escassez de dados paralelos;
* a ausência de padronização completa de acentos;
* a variabilidade morfológica e ortográfica;
* as limitações computacionais do treinamento em CPU;
* os comportamentos degenerados observados após o fine-tuning.

## Tecnologias

* Python
* Hugging Face Transformers
* NLLB-200
* PEFT / LoRA
* Seq2SeqTrainer
* sacreBLEU
* Pandas
* Jupyter Notebook

## Referência

Projeto desenvolvido como **EP2 — Tradução Automática de Baixo Recurso**, investigando tradução automática entre Português e Tupi Antigo por meio de um modelo multilíngue pré-treinado e adaptação eficiente com LoRA.
