import cv2
import winsound
from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO('C:/Users/Administrador/Desktop/TCC_OFICIAL_I.A/vigia_py/runs/detect/meu_treino7/weights/best.pt')

# Definir um limiar de confiança
CONFIDENCE_THRESHOLD = 0.82

# Abre a câmera 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

# Loop para capturar quadros da câmera
while True:
    # Lê um frame da câmera
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar imagem da câmera.")
        break

    # Executa a detecção de objetos no frame
    results = model(frame)  # A detecção ocorre aqui, usando YOLO

    # Processa os resultados de detecção
    for result in results:  # results contém uma lista de objetos detectados
        for obj in result.boxes.data:  # 'boxes' é uma estrutura que contém as caixas de detecção
            # A caixa de detecção contém os valores: (x1, y1, x2, y2, confianca, classe)
            x1, y1, x2, y2, conf, cls = obj.tolist()  # Converte o tensor para lista

            if conf >= CONFIDENCE_THRESHOLD:  # Verifica a confiança
                # Se a classe for a do tipo "arma", "faca", ou "pessoa"
                if int(cls) == 2:  # Classe 0: Arma
                    label = "ARMA DETECTADA"
                    color = (0, 0, 255)  # Vermelho
                    winsound.Beep(2000, 500)  # Alarme sonoro para arma

                elif int(cls) == 0:  # Classe 1: Faca
                    label = "FACA DETECTADA"
                    color = (0, 255, 255)  # Amarelo
                    winsound.Beep(1500, 500)  # Alarme sonoro para faca
                
                elif int(cls) == 1:  # Classe 2: Pessoa
                    label = "PESSOA DETECTADA"
                    color = (255, 0, 0)  # Azul

                # Desenha o retângulo e o texto na imagem
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibe o frame com as detecções
    cv2.imshow("Câmera ao Vivo", frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fecha a câmera e a janela
cap.release()
cv2.destroyAllWindows()
