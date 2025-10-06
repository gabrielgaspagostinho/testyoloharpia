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
    )

    print("Treinamento concluído!")
    print(f"Resultados salvos em: {results.save_dir}")

if __name__ == '__main__':

    main()
