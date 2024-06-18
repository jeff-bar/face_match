from pydantic import BaseModel
from typing import Optional

class DocumentNumber(BaseModel):
    document_number:str


class Data(DocumentNumber):
    name: Optional[str] = None
    origin: str
    img_path: Optional[str] = None
    dt_scanning: Optional[str] = None


