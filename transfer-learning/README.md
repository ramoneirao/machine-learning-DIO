# Desafio de Machine Learning: Transfer Learning com Deep Learning

Projeto desenvolvido para o desafio de Machine Learning da DIO (Digital Innovation One). O objetivo principal e aplicar a tecnica de **Transfer Learning** em uma rede neural profunda para resolucao de uma tarefa de classificacao de imagens, utilizando Python e TensorFlow/Keras no ambiente Google Colab ou localmente com o gerenciador **uv**.

---

## 1. Visao Geral do Projeto

O aprendizado por transferencia (Transfer Learning) consiste em reaproveitar o conhecimento adquirido por um modelo previamente treinado em uma base de dados massiva (como a ImageNet, contendo mais de 1 milhao de imagens e 1.000 classes) para uma nova tarefa com menor volume de dados ou recursos computacionais restritos.

Neste modulo, utilizamos a arquitetura convolucional **MobileNetV2** pre-treinada para classificar imagens do dataset **Cats vs Dogs** (dividido entre gatos e cachorros). Toda a estrutura e modular e pode ser reaproveitada para qualquer outro conjunto de dados binario ou multiclasse.

---

## 2. Estrutura do Diretorio

- `transfer_learning_cats_vs_dogs.ipynb`: Notebook Jupyter contendo o pipeline completo executado, visualizacoes, treinamento, graficos e inferencia.
- `README.md`: Documentacao tecnica do modulo.

---

## 3. Metodologia e Pipeline Desenvolvido

1. **Configuracao do Ambiente**: Deteccao de GPU e fixacao de seeds para reprodutibilidade.
2. **Carregamento dos Dados**: Dataset `cats_vs_dogs` particionado em 80% treino, 10% validacao e 10% teste via TFDS.
3. **Data Augmentation e Pre-processamento**: RandomFlip, RandomRotation, RandomZoom e padronizacao para 160x160 com `AUTOTUNE`.
4. **Arquitetura**: MobileNetV2 base com pesos congelados, GlobalAveragePooling2D, Dropout(0.2) e Dense(1, activation='sigmoid').
5. **Fase 1 (Feature Extraction)**: Treinamento com Adam (`1e-4`), EarlyStopping e ReduceLROnPlateau.
6. **Fase 2 (Fine-Tuning)**: Descongelamento a partir da camada 100 com learning rate reduzido (`1e-5`).
7. **Avaliacao**: Classification Report e Matriz de Confusao no conjunto de teste.
8. **Inferencia**: Funcao utilitaria para predicao de imagens locais ou remotas via URL.
9. **Exportacao**: Salvamento no formato `.keras`.

---

## 4. Como Executar com `uv`

Na raiz do repositorio:

```bash
# Sincronizar o ambiente virtual
uv sync

# Executar o Jupyter Lab com as dependencias do projeto
uv run --with jupyter jupyter lab
```

Abra o arquivo `transfer-learning/transfer_learning_cats_vs_dogs.ipynb` e selecione o kernel `.venv`.

---

## 5. Como Executar no Google Colab

1. Abra o [Google Colab](https://colab.research.google.com/).
2. Carregue o notebook `transfer_learning_cats_vs_dogs.ipynb`.
3. Altere o tipo de ambiente para **T4 GPU**.
4. Execute as celulas sequencialmente.