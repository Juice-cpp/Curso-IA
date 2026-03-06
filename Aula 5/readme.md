# Módulo 5: Visão Computacional

Este README foi ajustado para ficar **parelho com o conteúdo e a ordem** do notebook `notebook.ipynb` (teoria + exemplos).

**Carga Horária:** 12 horas  
**Nível:** Graduação/Pós-graduação  
**Pré-requisitos:** Módulos anteriores (Python, ML, Deep Learning básico). Conhecimentos de redes neurais e TensorFlow/PyTorch são desejáveis.

---

## 📦 O que tem nesta pasta

- `notebook.ipynb`: aula (teoria + exemplos)
- `AWS-1/`: dataset exportado do Roboflow em formato YOLOv8 (train/valid/test)
- `yolov8n.pt`, `yolov8l.pt`, `yolov8s-seg.pt`: pesos pré-treinados do YOLO
- `image.jpg`, `dog.jpeg`: imagens usadas em exemplos

---

## 📋 Objetivos (iguais ao notebook)

Ao final deste módulo, você será capaz de:

- Compreender o que é Visão Computacional e suas aplicações
- Aplicar operações comuns de processamento de imagens com OpenCV
- Entender os blocos básicos de uma CNN (convolução e pooling)
- Aplicar Transfer Learning com modelos pré-treinados
- Entender métricas de avaliação (classificação e detecção)
- Usar Roboflow para organizar/baixar datasets
- Treinar/validar/inferir com YOLOv8 (Ultralytics)

---

## 📚 Sumário (conteúdo do notebook)

1. O que é Visão Computacional
2. Processamento de imagens com OpenCV
3. CNN (Rede Neural Convolucional)
4. Transfer Learning
5. Avaliação de modelos
6. Roboflow
7. Detecção de objetos com YOLO (YOLOv8)
8. Atividade Prática — MNIST (classificação)

---

## 1️⃣ O que é Visão Computacional

**Visão Computacional** é um campo da IA que capacita máquinas a interpretar e entender o mundo visual a partir de imagens e vídeos. O objetivo é replicar a capacidade humana de reconhecer objetos, cenas e extrair informações.

**Relação com outros campos:**

- **Processamento de imagens:** transformações para melhorar qualidade/extrair características
- **Machine Learning:** algoritmos que aprendem a partir de dados visuais
- **Deep Learning:** CNNs (redes neurais convolucionais) que revolucionaram a área

### Aplicações

| Área | Exemplos |
|------|----------|
| **Saúde** | Diagnóstico por imagem (raios-X, tomografias), detecção de tumores |
| **Automotiva** | Carros autônomos, detecção de pedestres, leitura de placas |
| **Segurança** | Reconhecimento facial, vigilância por vídeo |
| **Varejo** | Checkout automático, análise de comportamento do cliente |
| **Agricultura** | Monitoramento de plantações, detecção de pragas |
| **Indústria** | Inspeção de qualidade, robótica |

### Desafios

- Variabilidade intraclasse (mesmo objeto com aparências diferentes)
- Condições de iluminação (sombras, reflexos)
- Oclusões (objetos parcialmente escondidos)
- Escala e rotação (tamanhos/orientações diferentes)
- Background complexo (fundo “poluído” atrapalha a detecção)

---

## 2️⃣ Processamento de imagens com OpenCV

O **OpenCV** é a biblioteca mais popular de visão computacional e reúne um grande conjunto de algoritmos para leitura, transformação, filtros e extração de informações.

### Instalação (no notebook)

```python
%pip install -q opencv-python matplotlib numpy
```

### Leitura e exibição de imagem

- `cv2.imread()` lê em **BGR**
- `cv2.cvtColor()` converte **BGR → RGB** para exibir corretamente no matplotlib

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()
```

### Redimensionamento

O `cv2.resize()` suporta diferentes interpolações:

- `cv2.INTER_LINEAR`: bilinear (boa opção geral)
- `cv2.INTER_CUBIC`: cúbica (melhor qualidade, mais lenta)
- `cv2.INTER_NEAREST`: vizinho mais próximo (rápida)

```python
img_resized = cv2.resize(img_rgb, (224, 224))
```

### Normalização e padronização

- **Normalização:** $[0,255] \to [0,1]$ dividindo por 255
- **Padronização:** subtrair média e dividir pelo desvio (ex: estatísticas do ImageNet)

```python
img_norm = img_rgb / 255.0

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_std = (img_norm - mean) / std
```

### Conversão de cores

- **RGB → Grayscale**: útil para bordas, limiarização, etc.
- **RGB → HSV**: útil para segmentação baseada em cor

```python
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
```

### Operações básicas (efeitos)

- **Gaussian Blur:** suaviza a imagem e reduz ruído
- **Threshold binário:** converte para dois níveis (0/255) com base em um limiar

```python
img_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
_, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
```

### Detecção de bordas (Canny)

```python
img_canny = cv2.Canny(img_gray, 100, 200)
```

### Transformações geométricas

No notebook há um exemplo de rotação usando matriz de transformação + `cv2.warpAffine()`.

```python
M = cv2.getRotationMatrix2D((224, 224), 45, 1)
img_rotated = cv2.warpAffine(img_rgb, M, (224, 224))
```

---

## 3️⃣ CNN (Rede Neural Convolucional)

Em imagens, pixels vizinhos costumam estar relacionados, formando padrões locais (bordas, texturas, formas). CNNs exploram essa estrutura; redes totalmente conectadas (densas) tendem a gerar muitos parâmetros e a ignorar a organização espacial.

Exemplo de ordem de grandeza (como no notebook): uma imagem 224×224×3 tem 150.528 valores. Se a primeira camada densa tiver 128 neurônios, isso dá ~19 milhões de pesos só nessa camada (fora vieses), aumentando custo computacional e risco de overfitting.

### Camadas convolucionais

Uma camada convolucional aplica **filtros (kernels)** sobre a imagem, gerando **mapas de características (feature maps)**.

Parâmetros principais (como no notebook):

- **Filters:** número de filtros (quantos mapas de características)
- **Stride:** passo do filtro (ex: 1 ou 2)
- **Padding:** se adiciona bordas para manter tamanho (ex: `same`) ou não (`valid`)

Exemplo rápido (TensorFlow/Keras):

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

conv_layer = Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    input_shape=(224, 224, 3),
)
```

### Exemplo didático: “detectar bordas”

O notebook também mostra um filtro clássico de borda com `cv2.filter2D()` (ideia de “primeiras camadas capturam bordas”).

### Pooling

Pooling reduz a resolução espacial e torna a rede mais robusta a pequenas variações.

- **MaxPooling:** pega o maior valor em uma janela
- **AveragePooling:** tira a média

```python
from tensorflow.keras.layers import MaxPooling2D
pool = MaxPooling2D(pool_size=(2, 2))
```

### Arquiteturas clássicas

| Arquitetura | Ano  | Características |
|------------|------|-----------------|
| **LeNet-5** | 1998 | CNN clássica para dígitos |
| **AlexNet** | 2012 | Marco no ImageNet (ReLU, GPUs) |
| **VGG**     | 2014 | Blocos repetidos com conv 3×3 |
| **ResNet**  | 2015 | Skip connections (mitiga vanishing gradient) |

### Exemplo de rede simples

Fluxo (como no notebook):

Imagem → Conv → Pooling → Conv → Pooling → Flatten → Dense → Classificação

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax"),
])
```

---

## 4️⃣ Transfer Learning

Treinar CNNs do zero exige muitos dados e custo computacional. **Transfer Learning** reaproveita modelos pré-treinados (ex: ImageNet), onde as primeiras camadas já aprenderam padrões gerais (bordas, texturas, cores, formas).

### Estratégias

- **Feature Extraction:** congela a base pré-treinada e treina apenas a “cabeça” final.
- **Fine-Tuning:** descongela algumas camadas superiores e treina com *learning rate* menor.

### Exemplo com TensorFlow/Keras (MobileNetV2)

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
predictions = layers.Dense(10, activation="softmax")(x)
model = models.Model(inputs=base_model.input, outputs=predictions)
```

---

## 5️⃣ Avaliação de modelos

No notebook aparecem métricas típicas:

### Classificação

- Acurácia, precisão, recall, F1-score
- Matriz de confusão
- Curva ROC (em cenários binários)

### Detecção/segmentação

- **IoU (Intersection over Union):**

$$
IoU = \frac{\text{Área da Interseção}}{\text{Área da União}}
$$

- **mAP (mean Average Precision):** métrica padrão em detecção
- **Curva Precision–Recall:** trade-off entre precisão e recall ao variar o limiar de confiança

---

## 6️⃣ Roboflow

Roboflow é uma plataforma para gerenciar datasets de visão computacional (anotação, versões, exportação).

🔗 https://roboflow.com

---

## 7️⃣ Detecção de objetos com YOLO (YOLOv8)

**YOLO** (“You Only Look Once”) trata detecção como um problema de regressão: em uma única passada a rede prevê caixas, classes e confiança.

### Instalação (Ultralytics)

```python
!pip install ultralytics
```

### Dataset deste repositório

O dataset está em `AWS-1/` (formato YOLOv8), com `train/`, `valid/` e `test/`.

> Observação importante (igual à ideia do notebook “lembra de alterar o arquivo data”):
> verifique o `AWS-1/data.yaml` antes de treinar. Para esta estrutura de pastas, os caminhos mais comuns são:

```yaml
train: train/images
val: valid/images
test: test/images
```

### Treinamento (API Python)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="AWS-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)
```

### Treinamento (CLI `yolo`)

O notebook também mostra o uso da CLI do Ultralytics via `!yolo ...`.

```python
!yolo task=detect mode=train model=yolov8l.pt data="AWS-1/data.yaml" epochs=2 imgsz=640 save_txt=true
```

### Validação e predição

```python
model.val()
model.predict(source="image.jpg", save=True, conf=0.25)
```

---

## 8️⃣ Atividade Prática — MNIST (Classificação)

### Objetivo

Treinar uma rede neural para **classificação de imagens no dataset MNIST**, experimentando diferentes arquiteturas, funções de ativação e otimizadores.

### Roteiro

1. Carregue o MNIST e normalize os dados (pixels de $[0, 255]$ para $[0, 1]$)
2. Divida em treino, validação e teste
   - **Treino:** 48.000 amostras
   - **Validação:** 12.000 amostras
   - **Teste:** 10.000 amostras
3. Construa pelo menos 3 modelos diferentes variando:
   - Número de camadas ocultas (1, 2 ou 3)
   - Número de neurônios por camada (ex: 32, 64, 128, 256)
   - Funções de ativação (ReLU, tanh, sigmoid)
4. Compile cada modelo com:
   - Otimizadores (SGD, Adam)
   - Taxas de aprendizado (0.01, 0.001, 0.0001)
   - Função de perda: `sparse_categorical_crossentropy`
   - Métrica: `accuracy`
5. Treine por até 20 épocas
   - Use *early stopping* se desejar
   - Monitore loss/accuracy de treino e validação
6. Avalie no conjunto de teste e reporte:
   - Acurácia final de cada modelo
   - Matriz de confusão do melhor modelo
   - Exemplos de erros (imagens classificadas incorretamente)
7. Compare e discuta:
   - Qual arquitetura performou melhor?
   - Qual otimizador convergiu mais rápido?
   - Houve overfitting? Como identificar?
   - Quais hiperparâmetros mais impactaram?

### Entrega

1. Notebook executado (`.ipynb`) com markdown explicando cada etapa
2. Gráficos obrigatórios:
   - Curvas de perda (treino e validação)
   - Curvas de acurácia (treino e validação)
   - Matriz de confusão do melhor modelo
3. Análise comparativa em markdown:
   - Tabela comparando os modelos
   - Discussão baseada nos resultados
4. Conclusão sobre as melhores escolhas

**Critérios de avaliação:**

- Clareza na documentação (markdown)
- Variedade de experimentos
- Qualidade das visualizações
- Profundidade da análise
- Conclusões baseadas em evidências

---

## 📝 Observações de execução

- O notebook mistura comandos de Jupyter/Colab (ex: `%pip`, `!yolo`). Em ambiente local, pode ser necessário adaptar caminhos e comandos.
- Para treinos do YOLO, uma GPU (ex: Google Colab com GPU) acelera bastante.

---

**Bons estudos!**

*Última atualização: Março 2026*