from enum import Enum
from urllib.parse import unquote, urlparse

from pydantic import BaseModel


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
    eml = "EML"
    msg = "MSG"

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
            "EML": ".eml",
            "MSG": ".msg",
        }
        return suffixes[self.value]


class File(BaseModel):
    url: str
    name: str | None = None

    @property
    def type(self) -> FileType | None:
        url = self.url
        if url:
            parsed_url = urlparse(url)
            path = unquote(parsed_url.path)
            extension = path.split(".")[-1].lower()
            try:
                return FileType[extension]
            except KeyError:
                raise ValueError(f"Unsupported file type for URL: {url}")
        return None

    @property
    def suffix(self) -> str:
        file_type = self.type
        if file_type is not None:
            return file_type.suffix()
        else:
            raise ValueError("File type is undefined, cannot determine suffix.")
