import cv2
import os
import random

# --- CONFIGURAÇÃO ---
# Caminhos para as pastas do seu dataset (relativos à pasta 'yoloteste')
DATASET_DIR = 'datasets'
# Nomes das classes, exatamente como no seu data.yaml
CLASS_NAMES = ['platform']
# Quantas imagens de cada conjunto (treino/val) você quer visualizar
NUM_SAMPLES = 10


def validate_set(image_dir, label_dir):
    """
    Função para visualizar anotações de um conjunto de dados (treino ou validação).
    """
    print(f"\n--- Validando o conjunto de dados em '{image_dir}' ---")

    # Pega uma amostra aleatória de imagens
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print(f"Nenhuma imagem encontrada em '{image_dir}'. Pulando.")
        return

    sample_images = random.sample(image_files, min(len(image_files), NUM_SAMPLES))

    for image_name in sample_images:
        # Monta os caminhos completos
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')

        # Carrega a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro ao carregar a imagem: {image_path}")
            continue

        h, w, _ = image.shape
        print(f"Verificando imagem: {image_name} (Dimensões: {w}x{h})")

        # Verifica se o arquivo de anotação existe
        if not os.path.exists(label_path):
            print(f"  -> AVISO: Arquivo de anotação não encontrado: {label_path}")
            cv2.imshow("Validacao do Dataset", image)
            if cv2.waitKey(0) == ord('q'): return
            continue

        # Lê o arquivo de anotação
        with open(label_path, 'r') as f:
            for line in f.readlines():
                try:
                    # Decodifica a linha do formato YOLO: class_id x_center y_center width height
                    class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line.split())

                    # Converte as coordenadas normalizadas de volta para pixels
                    x_center = x_center_norm * w
                    y_center = y_center_norm * h
                    box_width = width_norm * w
                    box_height = height_norm * h

                    # Calcula os cantos (x1, y1) e (x2, y2) para desenhar o retângulo
                    x1 = int(x_center - (box_width / 2))
                    y1 = int(y_center - (box_height / 2))
                    x2 = int(x_center + (box_width / 2))
                    y2 = int(y_center + (box_height / 2))

                    # Desenha a caixa e o rótulo na imagem
                    class_name = CLASS_NAMES[int(class_id)]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"  -> Caixa desenhada para a classe '{class_name}'")

                except Exception as e:
                    print(f"  -> ERRO ao processar a linha '{line.strip()}': {e}")

        # Mostra a imagem com as anotações
        cv2.imshow("Validacao do Dataset (Pressione 'q' para sair, qualquer outra tecla para proxima)", image)
        if cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    train_images_dir = os.path.join(DATASET_DIR, 'images/train')
    train_labels_dir = os.path.join(DATASET_DIR, 'labels/train')
    val_images_dir = os.path.join(DATASET_DIR, 'images/val')
    val_labels_dir = os.path.join(DATASET_DIR, 'labels/val')

    validate_set(train_images_dir, train_labels_dir)
    validate_set(val_images_dir, val_labels_dir)
    print("\nValidação concluída.")