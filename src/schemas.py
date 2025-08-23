from __future__ import annotations

from typing import Optional, List
from datetime import date
from pydantic import BaseModel, validator


class DocumentMetadata(BaseModel):
    """
    Pydantic model representing the classifier's output.

    Fields:
      - title: Concise, meaningful title (string)
      - correspondent: Shortest sender/institution name or None
      - tags: Up to 3 high-confidence tags (may be empty)
      - document_date: YYYY-MM-DD or None
      - document_type: Standard document type or None
      - language: Language code ("de", "en") or "und"
    """

    title: str
    correspondent: Optional[str]
    tags: List[str]
    document_date: Optional[date]
    document_type: Optional[str]
    language: str

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True
        schema_extra = {
            "example": {
                "title": "Rechnung 2024-1055",
                "correspondent": "Amazon",
                "tags": ["Rechnung", "Online-Kauf"],
                "document_date": "2024-05-25",
                "document_type": "Invoice",
                "language": "de",
            }
        }

    @validator("tags", pre=True, always=True)
    def ensure_tags_list(cls, v):
        # Accept None or single string and convert to list
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @validator("document_date", pre=True)
    def parse_document_date(cls, v):
        # Accept None, date or ISO-8601 string (YYYY-MM-DD)
        if v is None or v == "":
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            # Expect YYYY-MM-DD; date.fromisoformat handles this format
            try:
                return date.fromisoformat(v)
            except Exception as exc:
                raise ValueError(f"document_date must be YYYY-MM-DD or null: {exc}") from exc
        raise ValueError("document_date must be a date, ISO string (YYYY-MM-DD) or null")
