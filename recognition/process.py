import os
from tqdm import tqdm
from glob import glob
import numpy as np
import cv2 as cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import array 
import conf
import concurrent.futures
import util
from request import Data
from PIL import Image
import database

"""
Inicialize o aplicativo de análise de faces, utilizando o modelo arcface_r100_v1
"""
#name_modelo="antelopev2"
#name_modelo="buffalo_l"
name_modelo="arcface_r100_v1"
app = FaceAnalysis(name=name_modelo, root='./', providers=['CPUExecutionProvider'] )
app.prepare(ctx_id=0, det_size=(640, 640))


"""
Descrição

    Este método é projetado para receber um objeto np.ndarray que representa uma
    imagem e retornar um embedding normalizado do rosto contido na imagem. 
    Um embedding é uma representação vetorial da imagem do rosto, útil para tarefas 
    como reconhecimento facial e verificação de identidade.

Funcionalidade

    Entrada:

        O método recebe uma imagem no formato np.ndarray.

    Processamento:

        O método utiliza uma técnica de detecção facial para localizar o rosto na imagem.
        Se um rosto for detectado, a imagem do rosto é extraída e processada.
        Um embedding é gerado a partir da imagem do rosto utilizando uma rede neural treinada 
        especificamente para reconhecimento facial. 
        O embedding gerado é então normalizado para garantir que ele tenha uma escala consistente.
    
    Saída:

        Se um rosto for encontrado na imagem, o método retorna o embedding normalizado como um array.
        Caso um rosto não seja encontrado na imagem, o método retorna None.
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def get_embedding(img:np.ndarray) -> array:

    faces = app.get(img)
    if len(faces) == 1:
        return faces[0].normed_embedding
    else:
        return None
        


def get_kps(img:np.ndarray) -> array:

    faces = app.get(img)
    if len(faces) == 1:
        return faces[0]['kps']
    elif len(faces) > 1:
        raise ValueError("More than one face was found in the image")
    else:
        raise ValueError("No face found")


"""
Descrição

    Este método é projetado para receber dois arrays de embedding e um valor de limiar (threshold). 
    O método calcula o produto interno (dot product) entre os embeddings e retorna um valor booleano. 
    Se o produto interno for maior ou igual ao threshold fornecido, o método retorna True, 
    caso contrário, retorna False.

Funcionalidade

    Entrada:

        Um arrays de embedding (representados como np.ndarray), que são vetores de 
        características de rostos.
        Um arrays ou list de embedding (representados como array), que são vetores de 
        características de rostos.
        considerar os rostos como correspondentes.
        
    Processamento:

        O método calcula o produto interno (dot product) entre os dois embeddings fornecidos.
        Compara o valor do produto interno com o threshold.

    Saída:

        Retorna True se o produto interno for maior ou igual ao threshold.
        Retorna False caso contrário.
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def is_verify(embedding1:np.array, embedding2:array) -> bool:

    _, score = get_score(embedding1, embedding2)
    return similarity(score)


"""
Descrição

    Este método é projetado para receber um valor de score e compará-lo com um limiar (threshold) predefinido na configuração. O método avalia se o score é maior ou igual ao threshold e retorna um valor booleano. Se o score for maior ou igual ao threshold fornecido, o método retorna True, caso contrário, retorna False.

Funcionalidade

    Entrada:

        Um valor de score (representado como um número float), que representa a similaridade calculada entre dois embeddings de rostos.
        
    Processamento:

        O método compara o valor do score com o threshold predefinido na configuração.
        
    Saída:

        Retorna True se o score for maior ou igual ao threshold.
        Retorna False caso contrário.
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def similarity(score) -> bool:
    print(score)
    return score >= conf.THRESHOLD

"""
Descrição

    Este método é projetado para receber dois arrays de embedding e um valor de limiar (threshold). 
    O método calcula o produto interno (dot product) entre os embeddings e retorna um valor booleano. 
    Se o produto interno for maior ou igual ao threshold fornecido, o método retorna True, 
    caso contrário, retorna False.

Funcionalidade

    Entrada:

        Um arrays de embedding (representados como np.ndarray), que são vetores de 
        características de rostos.
        Um arrays ou list de embedding (representados como array ou list), que são vetores de 
        características de rostos.
        considerar os rostos como correspondentes.
        
    Processamento:

        O método calcula o produto interno (dot product) entre os dois embeddings fornecidos.
        Compara o valor do produto interno com o threshold.

    Saída:

        Retorna o maior score e seu index
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def get_score(embedding1:np.array, embedding2:array):

    if( isinstance( embedding2, np.ndarray) == False):
        embedding2 = np.array(embedding2)

    scores = np.dot(embedding1, embedding2.T)
    scores = np.clip(scores, 0., 1.)
    idx = np.argmax(scores)
    return idx, scores


"""
Descrição

    Este método é projetado para receber um número de documento e uma imagem no formato np.ndarray, 
    recuperar o documento associado ao número de documento no banco de dados e analisar se a 
    imagem fornecida corresponde à imagem armazenada no documento do banco de dados. 
    A verificação é feita através da comparação dos embeddings das duas imagens.

Funcionalidade

    Entrada:

        Um número de documento (como uma string).
        Uma imagem no formato np.ndarray.

    Processamento:

        O método se conecta ao banco de dados e recupera o documento associado ao número de 
        documento fornecido.
        Extrai a imagem armazenada no banco de dados e a converte em um formato apropriado (np.ndarray).
        Gera embeddings para a imagem fornecida e para a imagem recuperada do banco de dados.
        Compara os embeddings utilizando o produto interno (dot product) para verificar a similaridade.
    
    Saída:

        Retorna True se as imagens forem consideradas similares.
        Retorna False caso contrário.
        Retorna False se o número de documento não for encontrado no banco de dados.
        Se ocorrer um erro, lança uma exceção apropriada.
"""    
def compare_document_number(document_number, img:np.ndarray) -> bool:

    try:
        document_number = util.hash_doc(document_number)
        doc_database = database.find_by_id_doc(document_number)
        if(doc_database is not None ):
            kps = get_kps( img )
            img = util.treatment_img( img, kps )
            doc_embedding = get_embedding(img)
            return is_verify( doc_embedding, doc_database['embedding'] )
        else:
            return False
    except Exception as ex:
        raise ex


"""
Descrição

    Este método é projetado para receber duas imagens no formato np.ndarray e comparar se ambas são 
    similares. A verificação é feita através da comparação dos embeddings das duas imagens.

Funcionalidade
    
    Entrada:

        Duas imagens no formato np.ndarray.

    Processamento:

        Gera embeddings para cada uma das imagens utilizando um modelo de reconhecimento facial.
        Compara os embeddings utilizando o produto interno (dot product) para verificar a similaridade.

    Saída:

        Retorna True se as imagens forem consideradas similares.
        Retorna False caso contrário.
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def compare_doc(img1:np.ndarray, img2:np.ndarray) -> bool:
    
    try:  

        with concurrent.futures.ThreadPoolExecutor() as executor:

            kps1 = get_kps( img1 )
            futuro1 = executor.submit(util.treatment_img, img1, kps1)
            kps2 = get_kps( img2 )
            futuro2 = executor.submit(util.treatment_img, img2, kps2)

            # Espera pelos resultados
            result1 = futuro1.result()
            result2 = futuro2.result()

        return is_verify( get_embedding(result1), get_embedding(result2) )
    except Exception as ex:
        raise ex


"""
Descrição

    Este método é projetado para receber uma imagem no formato np.ndarray, pesquisar no 
    banco de dados por um documento cuja imagem seja similar à fornecida, e retornar o documento associado se a similaridade for aceita. Caso contrário, retorna None.

Funcionalidade

    Entrada:

        Uma imagem no formato np.ndarray.

    Processamento:

        O método se conecta ao banco de dados.
        Recupera todas as imagens armazenadas no banco de dados.
        Gera embeddings para a imagem fornecida e para as imagens recuperadas do banco de dados.
        Compara os embeddings utilizando o produto interno (dot product) para verificar a similaridade.
        Se a similaridade for aceita, retorna o documento associado.
        Caso contrário, retorna None.

    Saída:

        Retorna o documento associado se a similaridade for aceita.
        Retorna None se não houver documentos com imagens similares.
        Se ocorrer um erro, lança uma exceção apropriada.
  
"""
def find_by_doc(img:np.ndarray):

    try:
        kps = get_kps( img )
        img = util.treatment_img( img, kps )
        doc_embedding = get_embedding(img)
        results = database.find_by_doc(doc_embedding)

        embeddings = [result['embedding'] for result in results]
        data = [result for result in results]
        
        idx, scores = get_score(doc_embedding, embeddings)
        max_score = scores[idx]

        if similarity(max_score):
            return _load_data(data[idx])

        raise ValueError("Document not found")
    except Exception as ex:
        raise ex


"""
Descrição

    Este método é projetado para receber um número de documento e retornar o documento associado 
    a esse número de documento no banco de dados. O método faz uma consulta ao banco de dados 
    para encontrar o documento correspondente ao número de documento fornecido.

Funcionalidade

    Entrada:

        Um número de documento (como uma string).

    Processamento:

        O método se conecta ao banco de dados.
        Realiza uma consulta para encontrar o documento associado ao número de documento fornecido.
        Se o documento for encontrado, ele é retornado.
        Se nenhum documento for encontrado, o método retorna None.
    
    Saída:

        Retorna o documento associado ao número de documento como um dicionário.
        Retorna None se o número de documento não for encontrado no banco de dados.
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def find_by_document_number(document_number):

    try:
        document_number = util.hash_doc(document_number)
        data = database.find_by_id_doc(document_number)

        if data is not None:
            return _load_data(data)
        
        raise ValueError("Document not found")
    except Exception as ex:
        raise ex


"""
Descrição

    Este método é projetado para carregar dados de um dicionário e extrair valores específicos, como número do documento, caminho da imagem, data de digitalização, nome e origem. Para cada chave correspondente no dicionário, o método verifica se ela está presente e, em caso afirmativo, atribui seu valor a uma variável. Finalmente, o método retorna os valores extraídos como uma tupla.

Funcionalidade

    Entrada:

        Um dicionário de dados contendo possíveis chaves: 'document_number', 'img_path', 'dt_scanning', 'name' e 'origin'.

    Processamento:

        O método verifica a presença de cada chave no dicionário de dados.
        Atribui o valor correspondente à chave a uma variável se a chave estiver presente.
    
    Saída:

        Retorna uma tupla contendo os valores: (document_number, img_path, dt_scanning, name, origin).
        Se alguma chave não estiver presente no dicionário, o valor correspondente na tupla será None.
"""
def _load_data(data):

    document_number = None
    img_path = None
    dt_scanning = None
    name = None
    origin = None

    if( 'document_number' in data ):
        document_number = data['document_number']

    if( 'img_path' in data ):
        img_path = data['img_path']

    if( 'dt_scanning' in data ):
        dt_scanning = data['dt_scanning']

    if( 'name' in data ):
        name = data['name']
    
    if( 'origin' in data ):
        origin = data['origin']

    return document_number, img_path, dt_scanning, name, origin


"""
Descrição

    Este método é projetado para receber um número de documento, uma imagem no formato np.ndarray, 
    o caminho da imagem, e uma data de escaneamento, gerar um ID para o documento e salvar esses 
    dados no banco de dados.

Funcionalidade

    Entrada:

        Um número de documento (como uma string ou número).
        Uma imagem no formato np.ndarray.
        Um caminho da imagem (como uma string).
        Uma data de escaneamento (como uma string no formato YYYY-MM-DD).

    Processamento:

        O método se conecta ao banco de dados.
        Converte a imagem np.ndarray para um formato binário adequado para armazenamento (BLOB).
        Gera um ID para o documento (este ID pode ser gerado automaticamente pelo banco de dados se 
        configurado como autoincremento).
        Insere os dados na tabela de documentos do banco de dados.
        Fecha a conexão ao banco de dados.

    Saída:

        Não retorna nenhum valor.
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def save(data: Data, img:np.array):

    try:
        if data.dt_scanning is None:
            data.dt_scanning = util.get_data_now()
        if data.name is None or data.name == "":
            data.name = None
        if data.origin is None or data.origin == "":
            data.origin = None
        
        kps = get_kps( img )
        img = util.treatment_img( img, kps )
        embedding_doc = get_embedding( img )
        id_doc = util.hash_doc( data.document_number )

        database.insert( id_doc, data.document_number, embedding_doc, 
            data.img_path, data.name, data.dt_scanning, data.origin )
    except Exception as ex:
        raise ex



"""
Descrição

    Este método é projetado para receber um número de documento e deletar o documento 
    associado a esse número no banco de dados. O método se conecta ao banco de dados, 
    executa uma consulta para deletar o documento correspondente e retorna um booleano 
    indicando o sucesso da operação.

Funcionalidade

    Entrada:

        Um número de documento (como uma string ou número).

    Processamento:

        O método se conecta ao banco de dados.
        Executa uma consulta SQL para deletar o documento associado ao número fornecido.
        Retorna True se a operação for bem-sucedida.
        Retorna False se a operação falhar ou se o documento não for encontrado.
    
    Saída:

        Não retorna nenhum valor.
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def delete( document_number ):

    try:
        database.delete( util.hash_doc( document_number ) )
    except Exception as ex:
        raise ex


"""
Descrição

    Este método é projetado para criar a estrutura de um banco de dados, incluindo a 
    criação de tabelas necessárias para armazenar documentos e informações associadas. 
    O método se conecta ao banco de dados e executa comandos SQL para criar as tabelas 
    com seus respectivos campos.

Funcionalidade

    Processamento:

        O método se conecta ao banco de dados.
        Executa comandos SQL para criar a tabela de documentos, incluindo os campos necessários 
        (ex.: id, número de documento, embedding imagem, camino da imagem e data do scanning.).
        Fecha a conexão ao banco de dados.

    Saída:

        Não retorna nenhum valor.
        Se ocorrer um erro, lança uma exceção apropriada.
"""
def create():

    try:
        database.create_index()
    except Exception as ex:
        raise ex


############ funcao ####################################

def classify_rotation(angle):
    if -45 <= angle <= 45:
        return 0  # 0 degrees
    elif 45 < angle <= 135:
        return 90  # 90 degrees
    elif angle > 135 or angle < -135:
        return 180  # 180 degrees
    else:
        return 270  # 270 degrees
        
        
def compute_angle(landmarks):
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return np.degrees(np.arctan2(dy, dx))


def rotate_image(img, angle):
    if angle == 90:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)  
    elif angle == 180:
        img = cv2.flip(img, -1)
    elif angle == 270:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)  
    return img
    
'''
def cropped_face_ins(image, i):
    
    faces = app.get(image)
    
    for idx, face_info in enumerate(faces):
        bbox = face_info.bbox.astype(int).flatten()
        x1, y1, x2, y2 = bbox
        cropped_face = image[y1:y2, x1:x2, :]

        cv2.imwrite(f'ins_{i}.jpg', cropped_face)


def cropped_face_MTCNN(image, v):
    
    from mtcnn import MTCNN
    # Initialize MTCNN
    detector = MTCNN()

    # Detect faces
    faces = detector.detect_faces(image)

    # Crop and save faces
    for i, face in enumerate(faces):
        bounding_box = face['box']  # [x, y, width, height]
        x, y, w, h = bounding_box
        cropped_face = image[y:y+h, x:x+w]
        cv2.imwrite(f'MTCNN_{v}.jpg', cropped_face)



def normalize_illumination(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

'''


def align_face(img):

    faces = app.get(img)
    if len(faces) == 1:

        landmarks = faces[0].kps

        left_eye = landmarks[0]
        right_eye = landmarks[1]

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        h, w = img.shape[:2]
        center = (w / 2, h / 2) 

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    else:
        return None




if __name__ == '__main__':


    #img_path1 = "../fotos/teste/ice_cube_1.jpg"
    #img_path2 = "../fotos/teste/ice_cube_3.jpg"

    img_path1 = "../fotos/teste/tiago_2.jfif"
    img_path2 = "../fotos/teste/tiago_1.png"
    
    #img_path1 = "../fotos/teste/gabriel_file.jpeg"
    #img_path2 = "../fotos/teste/gabriel_1.png"

    #img_path1 = "../fotos/teste/walter_1.jpeg"
    #img_path2 = "../fotos/teste/walter_3.png"

    #img_path1 = "../fotos/teste/marcelo_1.jfif"
    #img_path2 = "../fotos/teste/marcelo_2.png"

    #img_path1 = "../fotos/teste/breno_file.jpeg"
    #img_path2 = "../fotos/teste/breno_1.png"
    
    #img_path1 = "../fotos/teste/gabriel_1_90.png"
    #img_path2 = "../fotos/teste/gabriel_1_270.png"
    #img_path3 = "../fotos/teste/gabriel_1_180.png"


    #img_path1 = "../fotos/teste/marcelo_2.png"
    #img_path2 = "../fotos/teste/marcelo_1.jfif"

    #cropped_face_ins(cv2.imread(img1), 1)
    #cropped_face_ins(cv2.imread(img2), 2)

    #cropped_face_MTCNN(cv2.imread(img1), 1)
    #cropped_face_MTCNN(cv2.imread(img2), 2)



    #image1 = normalize_illumination(image1)
    #image2 = normalize_illumination(image2)

    #image1 = cv2.imread(img_path1)
    #image2 = cv2.imread(img_path2)

    image1 = util.to_array( cv2.imread(img_path1), img_path1)
    image2 = util.to_array( cv2.imread(img_path2), img_path2)
    #image3 = util.to_array( cv2.imread(img_path3), img_path3)

    #image2 = normalize_illumination(image2)

    image1 = align_face(image1)
    image2 = align_face(image2)

    cv2.imwrite(f'image1.jpg', image1)
    cv2.imwrite(f'image2.jpg', image2)

    '''
    faces1 = app.get(image1)
    angle = compute_angle(faces1[0]['kps'])
    rotation = classify_rotation(angle)
    rotate_image1 = rotate_image(image1, rotation)

    faces2 = app.get(image2)
    angle = compute_angle(faces2[0]['kps'])
    rotation = classify_rotation(angle)
    rotate_image2 = rotate_image(image2, rotation)

    faces3 = app.get(image3)
    angle = compute_angle(faces3[0]['kps'])
    rotation = classify_rotation(angle)
    rotate_image3 = rotate_image(image3, rotation)
    
    cv2.imwrite(f'image1.jpg', rotate_image1)
    cv2.imwrite(f'image2.jpg', rotate_image2)
    cv2.imwrite(f'image3.jpg', rotate_image3)
    '''

    embedding1 = get_embedding(image1)
    embedding2 = get_embedding(image2)

    if is_verify(embedding1, embedding2):
        print( f"Face recognized" )
    else:
        print( f"Face not" )
    