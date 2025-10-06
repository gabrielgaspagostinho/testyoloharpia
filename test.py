from ultralytics import YOLO
import cv2
import sys

def main():
    MODEL_PATH = 'runs/detect/yolov8n_platform_detector6/weights/best.pt'
    IMAGE_PATH = 'datasets/images/train/115724ef-WIN_20251003_10_23_50_Pro.jpg'
    CONFIDENCE_THRESHOLD = 0.01 # Aumente este valor se tiver muitos falsos positivos


    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        sys.exit(1)


    frame = cv2.imread(IMAGE_PATH)

    # Verifica se a imagem foi carregada com sucesso
    if frame is None:
        print(f"Erro: Não foi possível carregar a imagem em '{IMAGE_PATH}'.")
        print("Verifique se o caminho do arquivo está correto e se o arquivo é uma imagem válida.")
        sys.exit(1)

    # Roda a inferência
    results = model.predict(IMAGE_PATH)

    print("--- INÍCIO DO DEBUG ---")
    print("Classes do modelo:", model.names)
    print("Shape da imagem:", frame.shape)
    
    # Processa e desenha os resultados na imagem
    for result in results:
        if len(result.boxes) == 0:
            print("O modelo não encontrou NENHUM objeto nesta imagem.")
        else:
            for box in result.boxes:
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]

                print(f"Objeto detectado: {class_name}, Confiança: {confidence:.2f}")

                if confidence > CONFIDENCE_THRESHOLD:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    label = f"{class_name} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print("--- FIM DO DEBUG ---")

    # Mostra a imagem final
    cv2.imshow("YOLOv8 Detection Test", frame)
    print("Pressione qualquer tecla na janela da imagem para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()