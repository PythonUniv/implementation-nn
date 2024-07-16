from pydantic import BaseModel, Field


class BeamTranslation(BaseModel):
    text: str
    probability: float = Field(..., ge=0, le=1)


class Translated(BaseModel):
    input: str
    output: str | list[BeamTranslation]
