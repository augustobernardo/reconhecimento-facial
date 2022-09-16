import cv2


# Esta variável vai controlar quantas fotos foram tiradas
amostra = 1

# Será tirada 25 fotos de cada pessoa (se for menor que 25 os algoritmos não vão conseguir aprender corretamente
numeroAmostras = 25

# Perguntando qual o nome do Aluno (identificador para cada imagem)
nomeAluno = input('Digite o nome do Aluno: ')

idAluno = input("Digite o id do aluno: ")

# Esta variável vai servir para controlar o qual que será o tamanho da foto que será tirada
largura, altura = 220, 220

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

camera = cv2.VideoCapture(0)

while True:
    _, imagem = camera.read()

    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    facesDetectadas = classificador.detectMultiScale(imagem,
                                                     scaleFactor=1.5,
                                                     minNeighbors=5,
                                                     minSize=(150, 150),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, largura, altura) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + largura, y + altura), (255, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Código do próprio OpenCV para redimensionar a imagem
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + 1], (largura, altura))

            # Salvando a foto em uma pasta/pessoa.(Indentificador).(NúmeroDaFoto).jpg, imagem redimensionada
            cv2.imwrite("fotos/" + str(nomeAluno) + "." + str(idAluno) + "." + str(amostra) + ".jpg", imagemFace)

            print("[foto" + str(amostra) + " capturada com sucesso]")
            # Incrementando o número da foto
            amostra += 1

            # Mostrando a imagem capturada da webcam
        cv2.imshow("Face", imagem)  # "Face" é o título da janela que será aberta

        # causa um delay para dar mais tempo para que a imagem seja processada e copiada para a tela durante a execução
        cv2.waitKey(1)

        if amostra >= numeroAmostras + 1:
            break

print("Fotos capturadas com sucesso!")

# Liberando a memória
camera.release()
cv2.destroyAllWindows()
