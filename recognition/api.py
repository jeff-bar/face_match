#!/usr/bin/env python
import logging 
from fastapi import FastAPI, status, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import process
from response import Response, MatchResponse, DocResponse
import numpy as np
from PIL import Image
import util
from request import Data, DocumentNumber
import time
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(filename='./log/api.log', 
    level=logging.ERROR, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('Rest API Facematch v2')
logger.info('INICIADO')

app = FastAPI(
    title="FaceMatch API",
    description="API for face matching and document management.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


async def load_image(doc: UploadFile) -> np.ndarray:
    """
    Load image file into a numpy array.
    """
    content_type = ["image/png","image/jpg","image/jpeg","image/jfif","application/pdf"]

    if doc.content_type in content_type:
        contents = await doc.read()
        return util.to_array(contents, doc.filename)
    else:
        logger.error(f'Error content type, allowed values {doc.filename} amount sent {doc.content_type}')
        raise HTTPException(status_code=500, detail=f"Error content type, allowed values {content_type}")


@app.get('/api/', response_model=Response, summary="Health Check", description="Check if the service is running.")
async def get() -> Response:
    return Response(code=0, message="Running v2")


@app.post('/api/find_by_doc', response_model=DocResponse, summary="Get Doc by Document", description="Retrieve a doc by its document.")
async def find_by_doc(foto: UploadFile = File(...)) -> DocResponse:
    
    try:
        image = await load_image(foto)
        document_number, img_path, dt_scanning, name, origin = process.find_by_doc(image)
        return DocResponse(code=0, message="Success", document_number=document_number, 
            img_path=img_path, dt_scanning=dt_scanning, name=name, origin=origin)

    except Exception as ex:
        erro = str(ex)
        logger.error(f'Error find {erro}')
        if 'Document not found' in erro:
            raise HTTPException(status_code=404, detail="Document not found")
        else:
            raise HTTPException(status_code=500, detail="Error retrieving document from repository")


@app.get('/api/find_by_document_number', response_model=DocResponse, summary="Get Doc by Document Number", description="Retrieve a doc by its document number.")
async def find_by_document_number(document_number: DocumentNumber = Depends()) -> DocResponse:
    
    try:
        document_number, img_path, dt_scanning, name, origin = process.find_by_document_number(document_number.document_number)
        return DocResponse(code=0, message="Success", document_number=document_number, 
            img_path=img_path, dt_scanning=dt_scanning, name=name, origin=origin)

    except Exception as ex:
        erro = str(ex)
        logger.error(f'Error find_by_document_number {document_number.document_number} {erro}')
        if 'Document not found' in erro:
            raise HTTPException(status_code=404, detail="Document not found")
        else:
            raise HTTPException(status_code=500, detail="Error retrieving document")


@app.post('/api/compare_document_number', response_model=MatchResponse, summary="Compare Document Number", description="Compare a document number with the image.")
async def compare_document_number(document_number: DocumentNumber = Depends(), foto: UploadFile = File(...)) -> MatchResponse:
    
    try:
        image = await load_image(foto)
        match = process.compare_document_number(document_number.document_number, image)
        return MatchResponse(code=0, message="Comparison made", match=match)

    except Exception as ex:
        erro = str(ex)
        logger.error(f'Error compare_by_document_number {document_number.document_number} {erro}')
        if 'Document not found' in erro:
            raise HTTPException(status_code=404, detail="Document not found")
        else:    
            raise HTTPException(status_code=500, detail="Error when searching and comparing document numbers in the document repository")


@app.post('/api/compare_doc', response_model=MatchResponse, summary="Compare Two Documents", description="Compare two documents by their images.")
async def compare_doc(foto1: UploadFile = File(...), foto2: UploadFile = File(...)) -> MatchResponse:
    
    try:
        image1 = await load_image(foto1)
        image2 = await load_image(foto2)
        match = process.compare_doc(image1, image2)
        return MatchResponse(code=0, message="Comparison made", match=match)

    except Exception as ex:
        erro = str(ex)
        logger.error(f'Error compare_doc {erro}')
        raise HTTPException(status_code=500, detail="Error comparing documents")


@app.post('/api/save_doc', response_model=Response, summary="Save Document", description="Save a document image to the repository.")
async def save_doc(data: Data = Depends(), doc: UploadFile = File(...)) -> Response:
    
    try:
        image = await load_image(doc)
        process.save(data, image)
        return Response(code=0, message="Document saved successfully")

    except Exception as ex:
        erro = str(ex)
        logger.error(f'Error save_doc document number {data.document_number} {erro}')
        raise HTTPException(status_code=500, detail="Error save document")


@app.delete('/api/delete', response_model=Response, summary="Delete Document", description="Delete a document by its number.")
async def delete(document_number: DocumentNumber = Depends()) -> Response:
    
    try:
        process.delete(document_number.document_number)
        return Response(code=0, message="Document successfully deleted")

    except Exception as ex:
        erro = str(ex)
        logger.error(f'Error delete document number {document_number.document_number} {erro}')
        if 'Document not found' in erro:
            raise HTTPException(status_code=404, detail="Document not found")
        else:
            raise HTTPException(status_code=500, detail="Error delete document")


@app.post('/api/create', response_model=Response, summary="Create Database", description="Create the database structure.")
async def create() -> Response:
    
    try:
        process.create()
        return Response(code=0, message="Creation of the table structure successfully completed") 

    except Exception as ex:
        erro = str(ex)
        logger.error(f'Error creating the database structure {erro}')
        raise HTTPException(status_code=500, detail="Error creating the database structure")
