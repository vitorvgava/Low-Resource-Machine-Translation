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

O fluxo experimental pode ser resumido como:

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

O pipeline completo inclui pré-processamento, inferência zero-shot, fine-tuning com LoRA, geração das traduções de teste, avaliação automática e seleção das dez melhores e dez piores sentenças por BLEU.

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
epochs       = 5
learning rate = 5e-5
weight decay = 0.01
batch size   = 4
early stopping patience = 2
```

A seleção final foi feita manualmente utilizando o **maior BLEU na validação**. Isso ocorreu porque o treinamento originalmente utilizou validation loss como critério de seleção automática, enquanto o BLEU era considerado a métrica mais relevante para a tarefa.

### Melhor checkpoint

Os resultados por época foram:

| Direção   | Melhor época |     BLEU |
| --------- | -----------: | -------: |
| PT → Tupi |            2 | 0.470439 |
| Tupi → PT |            2 | 0.988228 |

Em ambos os casos, a época com maior BLEU não correspondeu à época de menor validation loss.

## Resultados

### Comparação entre regimes

| Direção   | Regime    |       BLEU |       chrF1 |       chrF3 |
| --------- | --------- | ---------: | ----------: | ----------: |
| PT → Tupi | Zero-shot |     0.1190 |     11.9806 |     13.2296 |
| PT → Tupi | Few-shot  | **0.4611** |      6.8467 |      9.1878 |
| Tupi → PT | Zero-shot |     0.2404 |     12.3135 |     12.0392 |
| Tupi → PT | Few-shot  | **0.8308** | **13.9667** | **13.8709** |

O fine-tuning aumentou substancialmente o BLEU nas duas direções:

* **PT → Tupi:** `0.1190 → 0.4611`
* **Tupi → PT:** `0.2404 → 0.8308`

Entretanto, os ganhos em chrF foram assimétricos: na direção PT → Tupi, chrF1 e chrF3 diminuíram, enquanto na direção Tupi → PT ambas as métricas aumentaram.

## Análise qualitativa

O modelo zero-shot consegue produzir algumas traduções plausíveis, especialmente em expressões curtas e frequentes, mas também apresenta traduções semanticamente desvinculadas do texto original.

Exemplos observados incluem:

```text
pé rupi → tape rupi
nde rera → nde réra
```

Por outro lado, o fine-tuning aumenta a sensibilidade ao vocabulário e às estruturas presentes no corpus. Porém, também aparecem sinais claros de instabilidade e sobreajuste, principalmente na forma de **repetições patológicas de morfemas e fragmentos fonológicos**.

Exemplo de comportamento degenerado:

```text
xe abangaíba
→ xe îandéîaîaîaîaîaîaîaîaîaîaîaîaîaîaî...
```

Esses resultados indicam que o modelo adaptado pode apresentar melhor aderência ao corpus em situações favoráveis, mas também pode gerar sequências repetitivas, truncadas ou linguisticamente vazias quando encontra entradas pouco frequentes ou mais longas.

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

A descrição acima corresponde ao catálogo de arquivos apresentado no relatório.

## Principais conclusões

Os experimentos mostram que:

* o **zero-shot** apresenta capacidade limitada para o par Português–Tupi;
* o **fine-tuning com LoRA** melhora significativamente o BLEU;
* a melhoria é especialmente forte na direção **Tupi → Português**;
* os efeitos sobre métricas baseadas em caracteres são diferentes entre as duas direções;
* o fine-tuning também introduz comportamentos degenerados, incluindo repetições e saídas linguisticamente incoerentes;
* a baixa quantidade de dados, o treinamento em CPU, a ausência de regularização mais forte e a variabilidade ortográfica do corpus são limitações importantes.

## Limitações

Os resultados devem ser interpretados considerando:

* o tamanho reduzido do corpus de treinamento;
* a escassez de dados paralelos Português–Tupi;
* a ausência de padronização completa de acentos;
* a variabilidade morfológica e ortográfica do corpus;
* as limitações computacionais que levaram ao treinamento em CPU;
* os comportamentos degenerados observados após o fine-tuning.

## Tecnologias utilizadas

* **Python**
* **Hugging Face Transformers**
* **NLLB-200**
* **PEFT / LoRA**
* **Seq2SeqTrainer**
* **sacreBLEU**
* **Pandas**
* **Jupyter Notebook**

## Referência

Este projeto foi desenvolvido como o **EP2 — Tradução Automática de Baixo Recurso**, investigando tradução automática entre Português e Tupi Antigo utilizando modelos multilíngues pré-treinados e adaptação eficiente por LoRA.
