import cv2
import numpy as np

# Único código para fazer uma conexão com a webcam

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml")

# Aqui começa a captura da imagem câmera integrada é igual a 0
camera = cv2.VideoCapture(0)  # como parâmetro será passado um valor inteiro indicando qual câmera usar
#camera = cv2.VideoCapture("frontFacingExample.mp4")  # Para acessar vídeo MP4

# Esta variável vai controlar quantas fotos foram tiradas
amostra = 1

# Será tirada 25 fotos de cada pessoa (se for menor que 25 os algoritmos não vão conseguir aprender corretamente
numeroAmostras = 25

# Perguntando qual o nome do Aluno (identificador para cada imagem)
nomeAluno = input('Digite o nome do Aluno: ')

idPessoa = input("Digite o id da pessoa: ")

# Esta variável vai servir para controlar o qual que será o tamanho da foto que será tirada
largura, altura = 220, 220

while True:
    conectado, imagem = camera.read()  # Fazendo a leitura da webcam

    # Convertendo a imagem na escala de cinza
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Passando as imagens que serão detectadas com a escala de cinza
    facesDetectadas = classificador.detectMultiScale(imagem,
                                                     # Parâmetro que vai indicar escala da imagem como será com webcam
                                                     # essa escala é um pouco maior
                                                     scaleFactor=1.5,
                                                     # Tamanho mínimo para fazer uma detecção de faces é de 100, 100
                                                     minSize=(150, 150))
    # Colocar o "quadrado" envolta da face
    # facesDetectadas possui uma matriz que possui a posição x e y de onde começa uma face,
    # a largura e a altura da mesma
    for (x, y, l, a) in facesDetectadas:  # l --> largura / a --> altura
        # Percorrendo o rosto e desenhando o retângulo no mesmo
        # (ponto inicial), (ponto final), (cor bgr), borda do quadrado
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 255, 0), 2)

        # Detectando se existem olhos no quadrado de detecção do rosto
        regiao = imagem[y:y + a, x:x + l]  # nesta variável está apenas os olhos
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)

        # Detectando os olhos
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        # ====================================================================== #
        # Se for detectado apenas uma face o código não irá entrar no for abaixo #
        # fazendo com que seja impossível a captura da foto                      #
        # ====================================================================== #

        # Obrigatório a detecção dos olhos, caso não exista, não será possível a captura da foto
        # Laço para percorrer os olhos e desenhar o retângulo nos mesmos
        #for (ox, oy, ol, oa) in olhosDetectados:
            # Desenhando o retângulo envolta do olho
            #cv2.rectangle(imagem, (ox, oy), (ox + l, oy + a), (0, 255, 0), 2)

            # Tirando a foto / salvando a foto
            # Primeiro comando é para esperar uma tecla e o segundo é um comendo em hexadecimal e logo após a tecla
            # desejada
        if cv2.waitKey(1) & 0xFF == ord('q'):

            # Verificando a lumonosidade

            # Tirando a média dos valores RGB
            #if np.average(imagemCinza) > 110:  # Valor vai de 0 a 255 --> muito pequeno == pouco iluminação

            # Código do próprio OpenCV para redimensionar a imagem
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))

            # Salvando a foto em uma pasta/pessoa.(Indentificador).(NúmeroDaFoto).jpg, imagem redimensionada
            cv2.imwrite("fotos/" + str(nomeAluno) + "." + str(idPessoa) + "." + str(amostra) + ".jpg", imagemFace)

            print("[foto " + str(amostra) + " capturada com sucesso]")
            # Incrementando o número da foto
            amostra += 1

    # Mostrando a imagem capturada da webcam
    cv2.imshow("Face", imagem)  # "Face" é o título da janela que será aberta

    # causa um delay para dar mais tempo para que a imagem seja processada e copiada para a tela durante a execução
    cv2.waitKey(1)  # 1 é o tempo dado em milisegundos para processar cada uma das imagens

    if amostra >= numeroAmostras + 1:
        break

print("Fotos capturadas com sucesso!")

# Liberando a memória
camera.release()
cv2.destroyAllWindows()
