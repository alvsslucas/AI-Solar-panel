
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# Caminhos
MODEL_PATH = r"C:/TCC-CODE/git\AI-Solar-panel/modelotucadas.keras"
IMAGE_FOLDER = r"C:/Users/lumax/OneDrive/Área de Trabalho/Faculdade/TCC/solar dust/pack1/Sujo"
IMG_SIZE = (224, 224)

# Carregar modelo
model = load_model(MODEL_PATH)

# Contadores
contador_limpo = 0
contador_sujo = 0

# Processar todas as imagens da pasta
for nome_arquivo in os.listdir(IMAGE_FOLDER):
    if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png', '.avif')):
        caminho_imagem = os.path.join(IMAGE_FOLDER, nome_arquivo)

        img = load_img(caminho_imagem, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)
        classe = "Limpo" if np.argmax(pred) == 0 else "Sujo"
        confianca = np.max(pred)

        # Incrementar contadores
        if classe == "Limpo":
            contador_limpo += 1
        else:
            contador_sujo += 1

        # Logar resultado no terminal
        print(f"Imagem: {nome_arquivo} → Classificação: {classe} (Confiança: {confianca * 100:.2f}%)")

# Logar resumo
print(f"Total de imagens classificadas como Limpo: {contador_limpo}")
print(f"Total de imagens classificadas como Sujo: {contador_sujo}")
