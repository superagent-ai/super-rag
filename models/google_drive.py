from pydantic import BaseModel, Field


class GoogleDrive(BaseModel):
    service_account_key: dict = Field(
        ..., description="The service account key for Google Drive API"
    )
    drive_id: str = Field(..., description="The ID of a File or Folder")
