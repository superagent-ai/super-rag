from enum import Enum

from pydantic import BaseModel


class FileType(Enum):
    pdf = "PDF"
    docx = "DOCX"
    txt = "TXT"
    pptx = "PPTX"
    md = "MARKDOWN"


class File(BaseModel):
    type: FileType
    url: str
