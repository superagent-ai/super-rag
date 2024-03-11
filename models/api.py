from pydantic import BaseModel
from typing import Optional


class ApiError(BaseModel):
    message: Optional[str]
