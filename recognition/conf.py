import os

THRESHOLD=float( os.environ.get('THRESHOLD', 0.55) )

ANALYSIS_DATABASE_RECORD_SIZE=int( os.environ.get('ANALYSIS_DATABASE_RECORD_SIZE', 5) )

IS_NORMALIZE_ROTATION=bool( os.environ.get('IS_NORMALIZE_ROTATION', 'true') )

IS_FACE_ALIGNMENT=bool( os.environ.get('IS_FACE_ALIGNMENT', 'true') )

IMAGE_TMP=os.environ.get('IMAGE_TMP','/app/tmp')

HOST_DATABASE=os.environ.get('HOST_DATABASE','192.168.15.5')
PORT_DATABASE=int(os.environ.get( 'PORT_DATABASE',9200))
INDEX_DATABASE=os.environ.get('INDEX_DATABASE','face_recognition')
SCHEME_DATABASE=os.environ.get('SCHEME_DATABASE','http')
USERNAME_DATABASE=os.environ.get('USERNAME_DATABASE','elastic')
PASSWORD_DATABASE=os.environ.get('PASSWORD_DATABASE','vYz2Zp9_OMGj6kYgaK3m')