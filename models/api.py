from typing import Optional

from pydantic import BaseModel


class ApiError(BaseModel):
    message: Optional[str]
