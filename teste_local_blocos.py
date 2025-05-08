# teste_local_dividido.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image

# Caminhos
MODEL_PATH = "C:/TCC-CODE/TCC_code/modelo_treinando_t1.keras"
IMAGE_FOLDER = "C:/TCC-CODE/imagens_teste"
IMG_SIZE = (299, 299)
GRID_SIZE = 3  # 3x3 = 9 blocos por imagem

# Carregar modelo
model = load_model(MODEL_PATH)

# Função para dividir imagem em blocos menores
def dividir_em_blocos(img, grid=3):
    width, height = img.size
    bloco_w, bloco_h = width // grid, height // grid
    blocos = []
    for i in range(grid):
        for j in range(grid):
            caixa = (j * bloco_w, i * bloco_h, (j + 1) * bloco_w, (i + 1) * bloco_h)
            bloco = img.crop(caixa)
            blocos.append(bloco.resize(IMG_SIZE))
    return blocos

# Processar todas as imagens da pasta
for nome_arquivo in os.listdir(IMAGE_FOLDER):
    if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
        caminho_imagem = os.path.join(IMAGE_FOLDER, nome_arquivo)
        img_original = Image.open(caminho_imagem).convert('RGB')

        blocos = dividir_em_blocos(img_original, GRID_SIZE)
        resultados = []

        for bloco in blocos:
            arr = img_to_array(bloco)
            arr = preprocess_input(arr)
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr, verbose=0)
            classe = np.argmax(pred)
            resultados.append(classe)

        # Contagem de classes
        clean = resultados.count(0)
        dusty = resultados.count(1)

        # Decisão baseada na maioria
        if dusty > clean:
            classe_final = "Dusty"
            confianca_final = dusty / len(resultados)
        else:
            classe_final = "Clean"
            confianca_final = clean / len(resultados)

        # Mostrar imagem e resultado final
        plt.imshow(img_original)
        plt.axis("off")
        plt.title(f"{nome_arquivo} → {classe_final} (Confiança: {confianca_final * 100:.1f}%)")
        plt.show(block=False)
        plt.pause(2)
        plt.close()