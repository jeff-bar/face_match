from pydantic import BaseModel
from typing import Optional

class Response(BaseModel):
    code: int
    message: str

class MatchResponse(Response):
    match: bool

class DocResponse(Response):
    document_number: str
    img_path: Optional[str] = None
    dt_scanning: Optional[str] = None
    name: Optional[str] = None
    origin: str
    
    
