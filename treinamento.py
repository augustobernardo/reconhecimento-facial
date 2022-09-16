import cv2
import os
import numpy as np

# ============================== #
# Treinamento de Classificadores #
# ============================== #

# Criando os classificadores ou algoritmos
eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


# Método responsável por percorrer toda a pasta de fotos
# e retornará os respectivos ids de cada aluno numa lista
# ex: id1 -> ft1, ft2, ft3...
def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]  # Vai mostrar o caminho de cada imagem

    # Guardará as faces de cada aluno
    faces = []
    idsAlunos = []  # guardará os "ids" de cada aluno

    # Esta variável possui todos os caminhos da imagem, no laço pegando cada caminho de cada vez
    for caminhoImagem in caminhos:
        # Lendo as imagens direto do diretório | passando o caminho da imagem como parâmetro
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)  # Obrigatório fazer a conversão para
        # escala de cinza novamente
        # Capturando o "id" de cada pessoa
        # Este código vai percorrer o nome da foto e vai pegar um valor posterior ao "."
        idAluno = int(os.path.split(caminhoImagem[6:21].split(".")[1])[1])  # Ex: pessoa.(Id).2.jpg #

        # TODO: Criar um form para conseguir utilizar nomes diferentes, no final do form dar o append na tabela de ids

        # Lista com os ID's
        idsAlunos.append(idAluno)

        # Lista com as imagens
        faces.append(imagemFace)

    return np.array(idsAlunos), faces


# Atribuindo o retorno da função às variáveis
ids, faces = getImagemComId()

print(ids)

print("Treinando...")

# ===================== #
# Fazendo o treinamento #
# ===================== #
eigenface.train(faces, ids)  # Faz parte da aprendizagem supervisionada | é preciso ter duas pessoas
eigenface.write('classificadorEign.yml')  # O parâmetro é o nome do arquivo que será salvo o classificador, responsável
                                            # pela identificação dos ‘ids’

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento Realizado")

getImagemComId()
