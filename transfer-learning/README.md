# Desafio de Machine Learning: Transfer Learning com Deep Learning

Projeto desenvolvido para o desafio de Machine Learning da DIO (Digital Innovation One). O objetivo principal e aplicar a tecnica de **Transfer Learning** em uma rede neural profunda para resolucao de uma tarefa de classificacao de imagens, utilizando Python e TensorFlow/Keras no ambiente Google Colab.

---

## 1. Visao Geral do Projeto

O aprendizado por transferencia (Transfer Learning) consiste em reaproveitar o conhecimento adquirido por um modelo previamente treinado em uma base de dados massiva (como a ImageNet, com mais de 1 milhao de imagens e 1.000 classes) para uma nova tarefa com menor quantidade de dados ou menor disponibilidade computacional.

Neste projeto, utilizamos a arquitetura convolucional **MobileNetV2** pre-treinada para classificar imagens do dataset **Cats vs Dogs** (dividido entre gatos e cachorros). A mesma arquitetura pode ser estendida para qualquer outro dataset binario ou multiclasse.

---

## 2. Estrutura do Repositorio

- `transfer_learning_cats_vs_dogs.ipynb`: Notebook Jupyter principal contendo todo o pipeline estruturado passo a passo, pronto para execucao no Google Colab ou ambiente Jupyter local.
- `specs.MD`: Especificacoes e instrucoes do desafio proposto pela DIO.
- `README.md`: Documentacao tecnica do projeto, arquitetura e instrucoes de reproducao.

---

## 3. Metodologia e Pipeline Desenvolvido

O fluxo de execucao do notebook foi estruturado nas seguintes etapas:

1. **Configuracao do Ambiente**:
   - Verificacao de suporte a aceleracao por GPU (CUDA/cuDNN).
   - Fixacao de sementes aleatorias (seed=42) para garantir reprodutibilidade.

2. **Carregamento e Divisao dos Dados**:
   - Utilizacao do dataset `cats_vs_dogs` via TensorFlow Datasets (TFDS).
   - Divisao em 80% treino (~18.610 amostras), 10% validacao (~2.326 amostras) e 10% teste (~2.326 amostras).

3. **Data Augmentation e Pre-processamento**:
   - Redimensionamento padronizado para 160x160 pixels.
   - Camadas de aumento de dados (flip horizontal aleatorio, rotacao e zoom) para atenuar riscos de sobreajuste (overfitting).
   - Pipelines eficientes utilizando `tf.data`, com paralelismo via `AUTOTUNE`, `batch` e `prefetch`.

4. **Construcao da Arquitetura**:
   - Base convolucional: **MobileNetV2** (sem as camadas finais de classificacao do ImageNet).
   - Congelamento dos pesos do modelo base (`base_model.trainable = False`).
   - Normalizacao de pixels no intervalo [-1, 1].
   - Adicao de `GlobalAveragePooling2D`, regularizacao com `Dropout(0.2)` e neuronio denso com ativacao Sigmoide para classificacao binaria.

5. **Treinamento Fase 1 (Feature Extraction)**:
   - Treinamento apenas das camadas adicionadas no topo.
   - Otimizador Adam (`lr=1e-4`), funcao de perda `BinaryCrossentropy`.
   - Callbacks: `EarlyStopping` (interrupcao por estagnacao da perda de validacao) e `ReduceLROnPlateau`.

6. **Treinamento Fase 2 (Fine-Tuning)**:
   - Descongelamento parcial das camadas mais profundas da MobileNetV2 (a partir da camada 100).
   - Recompilacao com taxa de aprendizado reduzida (`lr=1e-5`) para ajuste fino dos filtros sem degradar o conhecimento previo.

7. **Avaliacao e Metricas de Desempenho**:
   - Avaliacao rigorosa no conjunto de teste independente.
   - Calculo de Acuracia, Precisao, Recall e F1-Score.
   - Geracao da Matriz de Confusao com Seaborn.

8. **Inferencia e Predicao em Novas Imagens**:
   - Funcao utilitaria capaz de receber uma URL publica ou arquivo local e predizer a classe com probabilidade associada.

9. **Exportacao do Modelo**:
   - Salvamento no formato padrao `.keras` (`transfer_learning_cats_vs_dogs.keras`).

---

## 4. Requisitos e Instalacao

Para executar o projeto localmente, instale as dependencias necessarias:

```bash
pip install tensorflow tensorflow-datasets numpy matplotlib seaborn scikit-learn
```

---

## 5. Como Executar no Google Colab

1. Acesse o [Google Colab](https://colab.research.google.com/).
2. Va em **Arquivo** > **Abrir notebook** > selecione a aba **GitHub** ou faca o upload de `transfer_learning_cats_vs_dogs.ipynb`.
3. Certifique-se de ativar o ambiente de GPU em **Ambiente de execucao** > **Alterar tipo de ambiente de execucao** > selecionar **T4 GPU**.
4. Execute as celulas sequencialmente (**Ambiente de execucao** > **Executar tudo**).

---

## 6. Resultados e Conclusoes

- O uso de Transfer Learning permitiu atingir acuracia superior a 95% no conjunto de teste de forma estavel e com baixo custo computacional.
- As tecnicas de Data Augmentation garantiram robustez a mudancas de escala, rotacao e espelhamento.
- O ajuste fino (Fine-Tuning) se mostrou eficaz ao ajustar os extratores de caracteristicas de alto nivel para distinguir caracteristicas sutis entre as classes.