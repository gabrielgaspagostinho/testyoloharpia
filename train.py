from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='datasets/data.yaml',
        epochs=100,
        imgsz=640,
        patience=100,
        name='yolov8n_platform_detector',
        batch=16,
        workers=4,
        # --- PARÂMETROS DE DATA AUGMENTATION ---
        degrees=15.0,  # Variação aleatória na rotação (em graus)
        translate=0.1,  # Variação aleatória na translação (fração da imagem)
        scale=0.1,  # Variação aleatória no zoom/escala
        fliplr=0.5,  # 50% de chance de espelhar a imagem horizontalmente
        hsv_h=0.015,  # Variação na Matiz (Hue) da cor
        hsv_s=0.7,  # Variação na Saturação da cor
        hsv_v=0.4  # Variação no Brilho (Value) da cor
    )

    print("Treinamento concluído!")
    print(f"Resultados salvos em: {results.save_dir}")

if __name__ == '__main__':
    main()