from enum import Enum
from urllib.parse import urlparse
from urllib.parse import unquote

from pydantic import BaseModel, validator


class FileType(Enum):
    pdf = "PDF"
    docx = "DOCX"
    txt = "TXT"
    pptx = "PPTX"
    md = "MARKDOWN"
    csv = "CSV"
    xlsx = "XLSX"
    html = "HTML"
    json = "JSON"

    def suffix(self) -> str:
        suffixes = {
            "TXT": ".txt",
            "PDF": ".pdf",
            "MARKDOWN": ".md",
            "DOCX": ".docx",
            "CSV": ".csv",
            "XLSX": ".xlsx",
            "PPTX": ".pptx",
            "HTML": ".html",
            "JSON": ".json",
        }
        return suffixes[self.value]


class File(BaseModel):
    url: str
    type: FileType | None = None

    @validator("type", pre=True, always=True)
    def set_type_from_url(cls, v, values):
        if v is not None:
            return v
        url = values.get("url")
        if url:
            parsed_url = urlparse(url)
            path = unquote(parsed_url.path)
            extension = path.split(".")[-1].lower()
            try:
                return FileType[extension]
            except KeyError:
                raise ValueError(f"Unsupported file type for URL: {url}")
        return v
