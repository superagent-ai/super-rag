from pydantic import BaseModel, Field


class AwsS3(BaseModel):
    remote_url: dict = Field(..., description="The remote URL of your public bucket")
