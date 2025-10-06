from ultralytics import YOLO
from PIL import Image  # Usaremos a biblioteca Pillow em vez do OpenCV para exibir
import sys
import os


def main():
    # --- CONFIGURAÇÃO ---
    MODEL_PATH = 'runs/detect/yolov8n_platform_detector6/weights/best.pt'
    IMAGE_PATH = 'imgteste2.jpg'

    # --- VERIFICAÇÃO DOS ARQUIVOS ---
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Arquivo do modelo não encontrado em '{MODEL_PATH}'")
        sys.exit(1)
    if not os.path.exists(IMAGE_PATH):
        print(f"ERRO: Arquivo de imagem não encontrado em '{IMAGE_PATH}'")
        sys.exit(1)

    # 1. Carrega o modelo treinado
    print(f"Carregando modelo de '{MODEL_PATH}'...")
    model = YOLO(MODEL_PATH)

    # 2. Roda a inferência diretamente no arquivo da imagem
    # A biblioteca ultralytics cuida de carregar a imagem para você.
    print(f"Processando imagem '{IMAGE_PATH}'...")
    results = model(IMAGE_PATH)  # Passamos o caminho diretamente

    # --- ANÁLISE E DEBUG NO TERMINAL ---
    # Esta parte é crucial para sabermos o que o modelo está "pensando"
    print("\n--- ANÁLISE DAS DETECÇÕES ---")

    # results é uma lista, pegamos o resultado da primeira (e única) imagem
    result = results[0]

    if len(result.boxes) == 0:
        print(">> O modelo não encontrou NENHUM objeto nesta imagem.")
    else:
        print(f">> O modelo encontrou {len(result.boxes)} objeto(s). Analisando:")
        for box in result.boxes:
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            print(f"  - Objeto: '{class_name}', Confiança: {confidence:.2f}")

    print("---------------------------\n")

    # 3. GERA A IMAGEM COM AS DETECÇÕES DESENHADAS
    # O método .plot() desenha automaticamente as caixas, classes e confianças.
    # Ele retorna um array NumPy da imagem no formato BGR.
    print("Gerando imagem de resultado...")
    annotated_frame = result.plot()

    # 4. EXIBE A IMAGEM USANDO PILLOW
    # Convertemos de BGR (padrão OpenCV/YOLO) para RGB (padrão Pillow)
    annotated_frame_rgb = annotated_frame[..., ::-1]
    img_result = Image.fromarray(annotated_frame_rgb)

    # .show() abre a imagem no seu visualizador de imagens padrão do sistema
    img_result.show()

    # Opcional: Salva a imagem com as detecções em um arquivo
    output_path = 'detection_result.jpg'
    img_result.save(output_path)
    print(f"Imagem com as detecções salva em: '{output_path}'")


if __name__ == '__main__':
    main()