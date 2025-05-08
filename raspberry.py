import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import time

# Caminho do modelo treinado
CAMINHO_MODELO = 'classificador_placa_solar.h5'
# Nome da imagem temporária
CAMINHO_IMAGEM = 'imagem_capturada.jpg'
# Tamanho esperado pela rede
TAMANHO_IMAGEM = (64, 64)


def tirar_foto():
    print("Capturando imagem da câmera...")
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # tempo para a câmera estabilizar
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Falha ao capturar imagem da câmera")

    cv2.imwrite(CAMINHO_IMAGEM, frame)
    print("Imagem salva como imagem_capturada.jpg")
    return CAMINHO_IMAGEM

def classificar_foto(modelo, caminho_imagem):
    img = load_img(caminho_imagem, target_size=TAMANHO_IMAGEM)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = modelo.predict(img_array)
    classe = "Clean" if np.argmax(pred) == 0 else "Dusty"
    confianca = np.max(pred)
    return classe, confianca


if __name__ == "__main__":
    print("Carregando modelo...")
    modelo = load_model(CAMINHO_MODELO)

    caminho = tirar_foto()
    classe, confianca = classificar_foto(modelo, caminho)

    print(f"Classificação: {classe} (confiança: {confianca:.2f})")
