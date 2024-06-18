
import logging 

logging.basicConfig(filename='log/api.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('Rest API Facematch UTIL v2')


from PIL import Image
import conf
import numpy as np
import hashlib
from datetime import datetime
import cv2
import fitz
import io
import os
import uuid
from pdf2image import convert_from_path

def to_array(img, file_name: str ="") -> np.ndarray:

    file_name = file_name.upper()

    if( isinstance( img, np.ndarray) == True):
        return img
    elif( file_name.endswith(".PDF")):
        return np.array( convert_pdf_to_jpeg(img) )
    else:
        nparr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def convert_pdf_to_jpeg(pdf_bytes):
    name = generate_name()
    name_pdf = name + '.pdf'
    name_jpeg = name + '.jpeg'
    
    try:
        save_tmp_pdf(pdf_bytes, name_pdf)
        img = convert_from_path(name_pdf, dpi=200)
        img[0].save(name_jpeg, 'JPEG')
        return cv2.imread(name_jpeg)
    finally:
        delete_tmp(name_pdf)
        delete_tmp(name_jpeg)
 

def generate_name():
    return conf.IMAGE_TMP + "/" + str(uuid.uuid4())

def save_tmp_pdf(pdf_bytes, name_pdf):
    pdf_doc = fitz.open('pdf', pdf_bytes)
    pdf_doc.save(name_pdf)
    pdf_doc.close()


def delete_tmp(name):
    try:
        if( os.path.exists(name) ):
            os.remove(name)
    except Exception as ex:
        logger.error("Erro ao deletar arquivo " + str(ex) )
    

def convert_to_jpeg(img, formato_saida='JPEG'):
    
    image = Image.open(io.BytesIO(img))

    imagem_jpg = image.convert('RGB')
    bytes_saida = io.BytesIO()
    imagem_jpg.save(bytes_saida, formato_saida)
    
    return bytes_saida.getvalue()


def hash_doc(id:str):

    key = ''.join(caractere for caractere in id if caractere.isdigit())

    hash_obj = hashlib.sha256( key.encode() )
    return hash_obj.hexdigest()


def treatment_img(img: np.ndarray, kps):
    
    if(conf.IS_NORMALIZE_ROTATION):
        angle = _compute_angle(kps)
        rotation = _classify_rotation(angle)
        img = _rotate_image(img, rotation)

    if(conf.IS_FACE_ALIGNMENT):
        img = _face_alignment(img, kps)

    return img


def _face_alignment(img, landmarks):

    if len(landmarks) > 0:

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



def _classify_rotation(angle):

    if -45 <= angle <= 45:
        return 0  # 0 degrees
    elif 45 < angle <= 135:
        return 90  # 90 degrees
    elif angle > 135 or angle < -135:
        return 180  # 180 degrees
    else:
        return 270  # 270 degrees
        
        
def _compute_angle(landmarks):

    if len(landmarks) > 0:

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    else:
        return None


def _rotate_image(img, rotation):

    if rotation == 90:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)  
    elif rotation == 180:
        img = cv2.flip(img, -1)
    elif rotation == 270:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)  
    return img



# Recuperando a data atual no formato %Y-%m-%dT%H:%M:%S
def get_data_now():
    
    today = datetime.now()
    return today.strftime("%Y-%m-%dT%H:%M:%S")



if __name__ == '__main__':

    img = "../fotos/teste/rafael_1.png"

    imagem = treatment_img( np.array( Image.open( img )) )    
    
    save_path = "filename_"+str(uuid.uuid4())+".jpg"

    #cv2.imwrite(save_path, imagem)

    im = Image.fromarray(imagem)
    im.save(save_path)

    # Substitua pdf_bytes pelos seus bytes de arquivo PDF
    #with open('/home/jefferson/Documentos/trabalho/facematch/foto/fotos_teste/pessoa1.pdf', 'rb') as pdf_file:
    #    pdf_bytes = pdf_file.read()

    #jpeg_images = convert_pdf_to_jpeg_in_memory(pdf_bytes)

    #im = Image.fromarray(jpeg_images)
    #im.save("./teste_.jpeg")
    # Agora, jpeg_images contém as imagens JPEG em memória (formato de bytes)