from pydantic import BaseModel, ConfigDict


class ExportedFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_name: str
    url: str


__all__ = ["ExportedFile"]
