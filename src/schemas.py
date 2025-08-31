from __future__ import annotations

from typing import Optional, List, Dict, Any
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


class Document(BaseModel):
    """
    Pydantic model representing a Paperless document in normalized form.

    - Tags are always plaintext List[str].
    - Correspondent and document_type are plaintext strings when available.
    - raw holds the original API response dict for reference.
    """

    id: Optional[int]
    title: Optional[str]
    tags: List[str]
    correspondent: Optional[str]
    document_date: Optional[date]
    document_type: Optional[str]
    language: Optional[str]
    raw: Optional[Dict[str, Any]] = None

    class Config:
        # Paperless API responses vary between installations; allow extra fields.
        extra = "allow"
        anystr_strip_whitespace = True
        schema_extra = {
            "example": {
                "id": 123,
                "title": "Rechnung 2024-1055",
                "correspondent": "Amazon",
                "tags": ["Rechnung", "Online-Kauf"],
                "document_date": "2024-05-25",
                "document_type": "Invoice",
                "language": "de",
            }
        }

    @validator("tags", pre=True, always=True)
    def normalize_tags(cls, v):
        """
        Normalize tags into a list of plaintext strings.

        Acceptable inputs:
         - None -> []
         - str -> [str]
         - list of strings -> as-is
         - list containing dicts (e.g. {'id': 1, 'name': 'foo'}) -> extract 'name' if present
         - ints -> converted to str (best-effort; callers should prefer name resolution upstream)
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        try:
            converted: List[str] = []
            for item in list(v):
                if isinstance(item, str):
                    converted.append(item)
                elif isinstance(item, dict):
                    # try to extract human-readable name fields
                    name = item.get("name") or item.get("title") or item.get("label")
                    if name:
                        converted.append(str(name))
                    else:
                        # fallback to stringified dict to avoid losing info
                        converted.append(str(item))
                elif isinstance(item, int):
                    converted.append(str(item))
                else:
                    converted.append(str(item))
            return converted
        except Exception:
            # Fallback: coerce to list of strings
            return [str(x) for x in v]

    @validator("document_date", pre=True)
    def parse_document_date(cls, v):
        # Accept None, date or ISO-8601 string (YYYY-MM-DD)
        if v is None or v == "":
            return None
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except Exception as exc:
                raise ValueError(f"document_date must be YYYY-MM-DD or null: {exc}") from exc
        raise ValueError("document_date must be a date, ISO string (YYYY-MM-DD) or null")

    def has_ai_processed_tag(self, ai_tag_name: str) -> bool:
        """
        Return True if the ai_tag_name appears in the document's tags (case-insensitive).
        """
        if not self.tags:
            return False
        try:
            target = ai_tag_name.strip().casefold()
            for t in self.tags:
                try:
                    if t and t.strip().casefold() == target:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    def model_dump_for_api(self) -> Dict[str, Any]:
        """
        Produce a dict suitable for PATCHing the Paperless API.

        Note: This returns plaintext values (tags as List[str], correspondent/name as str).
        PaperlessClient.update_document_from_metadata and other helpers are responsible
        for converting names into IDs when the API requires numeric ids.
        """
        payload: Dict[str, Any] = {}
        if self.title:
            payload["title"] = self.title
        if self.correspondent:
            payload["correspondent"] = self.correspondent
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.document_type:
            payload["document_type"] = self.document_type
        if self.document_date:
            try:
                payload["document_date"] = self.document_date.isoformat()  # type: ignore[attr-defined]
            except Exception:
                payload["document_date"] = str(self.document_date)
        return payload


class Tag(BaseModel):
    """
    Pydantic model representing a Paperless tag.

    Fields mirror the Paperless tag JSON structure example provided by the user:
      - id: numeric id of the tag
      - slug: short machine-readable slug
      - name: human-readable name
      - color: hex color string (e.g. "#a6cee3") or None
      - text_color: hex color string for text contrast or None
      - match: optional matching pattern (string) or None
      - matching_algorithm: numeric id of matching algorithm used
      - is_insensitive: case-insensitive matching flag
      - is_inbox_tag: whether this tag is an "inbox" tag
      - document_count: number of documents carrying this tag
      - owner: optional owner id
      - user_can_change: whether the current user can change the tag
    """

    id: int
    slug: str
    name: str
    color: Optional[str] = None
    text_color: Optional[str] = None
    match: Optional[str] = None
    matching_algorithm: Optional[int] = None
    is_insensitive: bool = False
    is_inbox_tag: bool = False
    document_count: int = 0
    owner: Optional[int] = None
    user_can_change: bool = False

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True
        schema_extra = {
            "example": {
                "id": 434,
                "slug": "ai_processed",
                "name": "AI_Processed",
                "color": "#a6cee3",
                "text_color": "#000000",
                "match": "",
                "matching_algorithm": 1,
                "is_insensitive": True,
                "is_inbox_tag": False,
                "document_count": 10,
                "owner": 3,
                "user_can_change": True,
            }
        }

    @validator("slug", "name", pre=True, always=True)
    def ensure_non_empty_str(cls, v):
        if v is None:
            raise ValueError("must not be null")
        return str(v).strip()

    @validator("match", pre=True, always=False)
    def empty_match_to_none(cls, v):
        # normalize empty strings to None for clarity
        if v is None or v == "":
            return None
        return str(v).strip()


class Correspondent(BaseModel):
    """
    Pydantic model representing a Paperless correspondent (sender / institution).

    Fields mirror the structure seen in the input examples:
      - id: numeric id of the correspondent
      - slug: short machine-readable slug
      - name: human-readable name
      - match: optional matching pattern (string) or None
      - matching_algorithm: numeric id of matching algorithm used
      - is_insensitive: case-insensitive matching flag
      - document_count: number of documents associated with this correspondent
      - owner: optional owner id
      - user_can_change: whether the current user can change the correspondent
    """

    id: int
    slug: str
    name: str
    match: Optional[str] = None
    matching_algorithm: Optional[int] = None
    is_insensitive: bool = False
    document_count: int = 0
    owner: Optional[int] = None
    user_can_change: bool = False

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True
        schema_extra = {
            "example": {
                "id": 158,
                "slug": "ammerlander-versicherung",
                "name": "Ammerländer Versicherung",
                "match": "",
                "matching_algorithm": 1,
                "is_insensitive": True,
                "document_count": 1,
                "owner": 3,
                "user_can_change": True,
            }
        }

    @validator("slug", "name", pre=True, always=True)
    def ensure_non_empty_str(cls, v):
        if v is None:
            raise ValueError("must not be null")
        return str(v).strip()

    @validator("match", pre=True, always=False)
    def empty_match_to_none(cls, v):
        # normalize empty strings to None for clarity
        if v is None or v == "":
            return None
        return str(v).strip()

  


class DocumentType(BaseModel):
    """
    Pydantic model representing a Paperless document type (docType).

    Fields mirror the structure seen in the input examples:
      - id: numeric id of the doc type
      - slug: short machine-readable slug
      - name: human-readable name
      - match: optional matching pattern (string) or None
      - matching_algorithm: numeric id of matching algorithm used
      - is_insensitive: case-insensitive matching flag
      - document_count: number of documents carrying this docType
      - owner: optional owner id
      - user_can_change: whether the current user can change the docType
    """

    id: int
    slug: str
    name: str
    match: Optional[str] = None
    matching_algorithm: Optional[int] = None
    is_insensitive: bool = False
    document_count: int = 0
    owner: Optional[int] = None
    user_can_change: bool = False

    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True
        schema_extra = {
            "example": {
                "id": 123,
                "slug": "entgeltabrechnung",
                "name": "Entgeltabrechnung",
                "match": "",
                "matching_algorithm": 1,
                "is_insensitive": True,
                "document_count": 1,
                "owner": 3,
                "user_can_change": True,
            }
        }

    @validator("slug", "name", pre=True, always=True)
    def ensure_non_empty_str(cls, v):
        if v is None:
            raise ValueError("must not be null")
        return str(v).strip()

    @validator("match", pre=True, always=False)
    def empty_match_to_none(cls, v):
        # normalize empty strings to None for clarity
        if v is None or v == "":
            return None
        return str(v).strip()
