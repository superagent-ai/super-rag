from enum import Enum
from pydantic import BaseModel


class FileType(Enum):
    pdf = "PDF"
    docx = "DOCX"
    txt = "TXT"
    pptx = "PPTX"
    csv = "CSV"
    xlsx = "XLSX"
    md = "MARKDOWN"


class File(BaseModel):
    type: FileType
    url: str
