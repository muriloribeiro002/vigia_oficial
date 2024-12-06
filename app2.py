
import subprocess
from twilio.rest import Client
from ultralytics import YOLO
import cv2
import winsound
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/executar_codigo', methods=['POST'])
def executar_codigo():
    try:
        
        account_sid = ''  
        auth_token = ''  

        
        client = Client(account_sid, auth_token)
   
        resultado = "Código Python executado com sucesso!"
        
       
        model = YOLO('C:/Users/Administrador/Desktop/TCC_OFICIAL_I.A/vigia_py/runs/detect/meu_treino7/weights/best.pt')


        CONFIDENCE_THRESHOLD = 0.82

      
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Erro ao acessar a câmera.")
            exit()

   
        while True:
           
            ret, frame = cap.read()
            if not ret:
                print("Falha ao capturar imagem da câmera.")
                break

           
            results = model(frame)

            
            for result in results:
                for obj in result.boxes:
                    
                    if obj.conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = map(int, obj.xyxy[0])  #

                       
                        if obj.cls == 2:  # Classe 0: Arma
                            label = "ARMA DETECTADA"
                            color = (0, 0, 255)  
                            winsound.Beep(2000, 500) 
                            # Enviar uma mensagem SMS
                            message = client.messages.create(
                                body='UMA ARMA FOI DETECTADA',  
                                from_='',  
                                to=''  
                            )

                            
                            print(f"Mensagem SID: {message.sid}")

                        if obj.cls == 0:  # Classe 1: Faca
                            label = "FACA DETECTADA"
                            color = (0, 255, 255)  
                            winsound.Beep(1500, 500) 
                                                        
                            message = client.messages.create(
                                body='UMA FACA FOI DETECTADA',  
                                from_='',  
                                to='' 
                            )

                        
                        if obj.cls == 1:  # Classe 1: Faca
                            label = "PESSOA DETECTADA"
                            color = (255, 0, 0)  
                            
                        

                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Mostra o frame na tela
            cv2.imshow("Câmera ao Vivo", frame)

            # Sai do loop se a tecla 'q' for pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Fecha a câmera e a janela
        cap.release()
        cv2.destroyAllWindows()



                
                

        return resultado
    except Exception as e:
        return f"Ocorreu um erro: {e}"

if __name__ == '__main__':
    app.run(debug=True)
