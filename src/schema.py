"""Pydantic models for the LLM output (same fields and values as GUIDELINES.md)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

SiNo = Literal["si", "no"]
SiNoNA = Literal["si", "no", "no_aplica"]


class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    gender: Literal["mujer", "hombre", "desconocido"]
    gender_evidence: str
    role: Literal["estudio", "experta_externa", "paciente_testimonio", "otro"]
    role_evidence: str
    leads_study: SiNoNA
    quotes: list[str]  # ids of the quotes numbered in the text ("C3")
    is_interviewee: bool
    quote_origin: Literal["a_sinc", "comunicado", "publicacion", "no_especificado", "no_aplica"]
    intro_style: Literal["con_credenciales", "solo_nombre", "no_aplica"]
    position: str
    affiliation: str
    honorific: Literal["doctor", "profesor", "otro", "ninguno"]
    later_surname: SiNoNA
    later_first_name: SiNoNA
    in_title_or_lead: SiNo

    @model_validator(mode="after")
    def check_identified(self) -> "Person":
        # the other dependencies between fields are normalised in extract.py (normalize_person)
        if not self.name.strip() and not self.description.strip():
            raise ValueError("persona sin name ni description")
        return self


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    people: list[Person]
    collective_quotes: list[str]  # ids
    non_quotes: list[str]  # ids of texts in quotation marks that are not quotes (terms, titles)
