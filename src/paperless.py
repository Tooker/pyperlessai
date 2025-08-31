import os
from typing import Optional, List, Dict, Any, Tuple, Union

import httpx
from loguru import logger
from schemas import DocumentMetadata, Document, Tag, DocumentType, Correspondent

class PaperlessClient:
    """
    Encapsulates communication with a paperless-ngx instance.

    Usage (preferred for long-running/bulk operations):
        async with PaperlessClient(base_url="https://paperless.example", api_token="TOKEN") as client:
            docs = await client.list_documents(limit=10)
            data = await client.fetch_document_bytes(doc_id)

    The client also remains usable without the context manager: methods will create and close
    temporary httpx.AsyncClient instances when no shared client is active.
    """

    def __init__(self, base_url: str, api_token: str, request_timeout: int = 30):
        if not base_url:
            raise ValueError("base_url is required")
        if not api_token:
            raise ValueError("api_token is required")

        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.request_timeout = request_timeout
        self.headers = {"Authorization": f"Token {self.api_token}"}
        # httpx.Timeout expects seconds for connect/read/write; use a reasonable connect timeout.
        self.timeout = httpx.Timeout(self.request_timeout, connect=10.0)

        # Shared AsyncClient when used as an async context manager
        self._client: Optional[httpx.AsyncClient] = None

        
    async def __aenter__(self) -> "PaperlessClient":
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    async def _acquire_client(self, external: Optional[httpx.AsyncClient] = None) -> Tuple[httpx.AsyncClient, bool]:
        """
        Decide which httpx.AsyncClient to use for an operation.

        Precedence:
          1) If a shared client (self._client) exists, use it and indicate it should NOT be closed.
          2) Else if an external client was supplied, use it and indicate it should NOT be closed.
          3) Otherwise create a temporary client and indicate it SHOULD be closed by the caller.
        Returns (client, should_close)
        """
        if self._client is not None:
            return self._client, False
        if external is not None:
            return external, False

        client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout)
        return client, True

    async def _updateInternalMaps(self):
        self._tags = await self.list_tags()
        self._docTypes = await self.list_document_types()
        self._correspondents = await self.list_correspondents()
        pass

    async def list_documents(
        self,
        client: Optional[httpx.AsyncClient] = None,
        limit: Optional[int] = None,
    ) -> List[Document]:
        """
        Return a list of Document models (normalized). Accepts an existing httpx.AsyncClient
        or will use the shared client when PaperlessClient is used as a context manager.
        If no client is available, a temporary client will be created and closed automatically.

        The method is tolerant of different API response shapes (DRF-style 'results'/'next' or direct lists)
        and converts raw document dicts into normalized `Document` instances with plaintext tags.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        await self._updateInternalMaps()

        docs: List[Dict[str, Any]] = []
        next_url = f"{self.base_url}/api/documents/"

        try:
            while next_url:
                logger.info(f"Fetching documents page: {next_url}")
                resp = await client.get(next_url)
                resp.raise_for_status()
                data = resp.json()
                page_results = data.get("results") or data.get("data") or data
                if isinstance(page_results, list):
                    docs.extend(page_results)
                elif isinstance(page_results, dict):
                    docs.append(page_results)
                else:
                    logger.warning(f"Unexpected documents response shape: {type(page_results)}")

                if limit and len(docs) >= limit:
                    # We'll still collect up to limit (models created below)
                    break

                if next_url := data.get("next"):
                    next_url = next_url.replace("http://","https://")
        finally:
            if created_client:
                await created_client.aclose()
        docs = await self._parseDocumentData(docs[:limit])

        return docs

    def _findIdInList(self, id_value: Union[int, str, Dict[str, Any]], listOf: Union[List[Tag], List[DocumentType], List[Correspondent]]) -> Optional[str]:
        """
        Given an id-like value or name and a list of Tag/DocumentType/Correspondent objects,
        return the human-readable name if found, otherwise None.

        Accepts:
          - int -> match by item.id
          - str -> either numeric id string or name; try id first then name (case-insensitive)
          - dict -> extract 'id' or 'pk' (or 'name'/'title'/'label') then match
        """
        try:
            if id_value is None:
                return None

            # If a dict-like object was passed, try to extract an id or a name
            if isinstance(id_value, dict):
                id_candidate = id_value.get("id") or id_value.get("pk")
                if id_candidate is not None:
                    id_value = id_candidate
                else:
                    # fallback to name-like fields
                    name_candidate = id_value.get("name") or id_value.get("title") or id_value.get("label")
                    return str(name_candidate).strip() if name_candidate else None

            # Try to interpret as an integer id first
            try:
                iv = int(id_value)
            except Exception:
                # Treat as a name; try to match case-insensitively against known items
                target = str(id_value).strip().casefold()
                for item in listOf or []:
                    try:
                        nm = getattr(item, "name", None)
                        if nm and nm.strip().casefold() == target:
                            return nm
                    except Exception:
                        continue
                # No exact match found; return the original string value (trimmed)
                return str(id_value).strip()
            else:
                # Match integer id against items' id attribute
                for item in listOf or []:
                    try:
                        if getattr(item, "id", None) == iv:
                            return getattr(item, "name", None)
                    except Exception:
                        continue
                return None
        except Exception:
            return None


    async def _parseDocumentData(self, response:List[Dict]):

        self._updateInternalMaps()
          # Convert raw document dicts into Document models with plaintext tag names
       

        normalized: List[Document] = []
        for d in response:
            try:

              
                # Extract tag-like fields from common locations
                raw_tags = d.get("tags") or []
                tag_names: List[str] = [ self._findIdInList(raw,self._tags) for raw in raw_tags]
                correspondent_name = self._findIdInList(d.get("correspondent"),self._correspondents)
                document_type_name = self._findIdInList(d.get("document_type"),self._docTypes)
                pdfData = await self.fetch_document_bytes(None,doc_id=d.get("id"))

                

                normalized_doc = Document.model_validate(
                    {
                        "id": d.get("id"),
                        "title": d.get("title"),
                        "tags": tag_names,
                        "correspondent": correspondent_name,
                        "document_date": d.get("document_date") or d.get("date"),
                        "document_type": document_type_name,
                        "language": d.get("language"),
                        "raw": d,
                        "pdfdata": pdfData
                    }
                )
                normalized.append(normalized_doc)
            except Exception:
                # Skip malformed document entries but continue processing others
                logger.exception(f"Failed to normalize document: {d}")
                continue

        
        return normalized




    async def fetch_document_bytes(
        self,
        client: Optional[httpx.AsyncClient],
        doc_id: int,
    ) -> Optional[bytes]:
        """
        Attempts to fetch a document's thumbnail, falling back to download.
        Accepts an httpx.AsyncClient instance (recommended) for backwards compatibility.
        When PaperlessClient is used as a context manager, the shared client will be used.
        Returns bytes (image/pdf) or None.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        try:
            thumb_url = f"{self.base_url}/api/documents/{doc_id}/thumbnail/"
            download_url = f"{self.base_url}/api/documents/{doc_id}/download/"
            urls = [thumb_url, download_url]

            for url in urls:
                try:
                    logger.debug(f"Trying {url}")
                    resp = await client.get(url)
                    # Some endpoints may return non-200 for not-found or redirects
                    if resp.status_code == 200 and resp.content:
                        ct = resp.headers.get("content-type", "")
                        if "application/json" in ct:
                            try:
                                j = resp.json()
                            except Exception:
                                logger.debug("JSON content-type but failed to parse JSON")
                                continue
                            file_url = j.get("url") or j.get("download_url")
                            if file_url:
                                logger.debug(f"Found file URL in JSON response, fetching {file_url}")
                                resp2 = await client.get(file_url)
                                resp2.raise_for_status()
                                return resp2.content
                            else:
                                logger.debug("JSON response but no file URL field found")
                                continue
                        # treat content as raw bytes (image/pdf)
                        return resp.content
                    else:
                        logger.debug(f"Non-200 ({resp.status_code}) when trying {url}")
                except httpx.HTTPStatusError as e:
                    logger.warning(f"HTTP error fetching {url}: {e}")
                except Exception as e:
                    logger.warning(f"Error fetching {url}: {e}")
        finally:
            if created_client:
                await created_client.aclose()

        return None


    async def _fetch_list(
        self,
        client: Optional[httpx.AsyncClient],
        endpoints: List[str],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generic helper to fetch paginated lists from one of multiple possible endpoints.
        Tries each endpoint in order until it finds one that returns results.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        try:
            for ep in endpoints:
                items: List[Dict[str, Any]] = []
                next_url = f"{self.base_url}/api/{ep}/"
                try:
                    while next_url:
                        next_url = next_url.replace("http://","https://")
                        logger.debug(f"Fetching {ep} page: {next_url}")
                        resp = await client.get(next_url)
                        # If endpoint not found, try the next candidate
                        if resp.status_code == 404:
                            logger.debug(f"Endpoint /api/{ep}/ returned 404, trying next endpoint")
                            items = []
                            break

                        resp.raise_for_status()
                        data = resp.json()
                        page_results = data.get("results") or data.get("data") or data
                        if isinstance(page_results, list):
                            items.extend(page_results)
                        elif isinstance(page_results, dict):
                            items.append(page_results)
                        else:
                            logger.warning(f"Unexpected response shape for /api/{ep}/: {type(page_results)}")

                        if limit and len(items) >= limit:
                            return items[:limit]

                        next_url = data.get("next")
                except httpx.HTTPStatusError as e:
                    logger.warning(f"HTTP error when fetching /api/{ep}/: {e}")
                    items = []
                except Exception as e:
                    logger.warning(f"Error when fetching /api/{ep}/: {e}")
                    items = []

                if items:
                    return items[:limit] if limit else items

            # If none of the endpoints returned data, return empty list
            return []
        finally:
            if created_client:
                await created_client.aclose()


    async def list_tags(
        self,
        client: Optional[httpx.AsyncClient] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of tag objects from the paperless API.
        Tries common tag endpoint paths.
        """
        endpoints = ["tags"]
        tagsasjson =  await self._fetch_list(client, endpoints, limit=limit)
        return [Tag(**tags) for tags in tagsasjson]



    async def list_document_types(
        self,
        client: Optional[httpx.AsyncClient] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of document type objects. Tries several common endpoint name variants.
        """
        endpoints = ["document_types"]
        docTypesjson = await self._fetch_list(client, endpoints, limit=limit)
        return [DocumentType(**doc) for doc in docTypesjson]


    async def list_correspondents(
        self,
        client: Optional[httpx.AsyncClient] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of correspondent/contact objects. Tries a few possible endpoint names.
        """
        endpoints = ["correspondents"]
        correspondentjson = await self._fetch_list(client, endpoints, limit=limit)
        return [Correspondent(**corr) for corr in correspondentjson]


    # Utilities to produce simplified id -> name maps and pretty-printed representations
    def _name_map_from_items(self, items: List[Dict[str, Any]]) -> Dict[int, str]:
        """
        Build a mapping of item id -> item name from a list of dict-like items.

        This helper is more tolerant about variations in API responses: some Paperless
        deployments return 'id', others 'pk'. Likewise the name field may be 'name'
        or 'title'. Accept both forms and skip malformed entries.
        """
        result: Dict[int, str] = {}
        for item in items:
            try:
                id_val = item.get("id") if isinstance(item, dict) else None
                if id_val is None:
                    id_val = item.get("pk") if isinstance(item, dict) else None

                # Accept a few possible name fields
                name_val = None
                if isinstance(item, dict):
                    name_val = item.get("name") or item.get("title") or item.get("label")

                if id_val is not None and name_val is not None:
                    try:
                        result[int(id_val)] = name_val
                    except Exception:
                        # skip items with non-int ids
                        continue
            except Exception:
                # Ignore malformed items
                continue
        # keep deterministic ordering by id
        return dict(sorted(result.items()))

    def _pretty_from_map(self, m: Dict[int, str]) -> str:
        """
        Return a stable, newline-separated pretty string for a mapping of id -> name.
        Sorted by id for determinism.
        """
        return "\n".join(f"{k}: {v}" for k, v in sorted(m.items()))

    async def tags_name_map(self, client: Optional[httpx.AsyncClient] = None) -> Dict[int, str]:
        """
        Return a simplified mapping of tag id -> tag name.
        """
        try:
            items = await self.list_tags(client=client)
            result = self._name_map_from_items(items)
            # Ensure deterministic ordering by id
            return dict(sorted(result.items()))
        except Exception as e:
            logger.exception("Failed to fetch tags: %s", e)
            return {}

    async def tags_pretty(self, client: Optional[httpx.AsyncClient] = None) -> str:
        """
        Return a pretty-printed string of tag id -> name pairs (one per line).
        """
        return self._pretty_from_map(await self.tags_name_map(client=client))

    async def document_types_name_map(self, client: Optional[httpx.AsyncClient] = None) -> Dict[int, str]:
        """
        Return a simplified mapping of document type id -> document type name.
        """
        try:
            items = await self.list_document_types(client=client)
            result = self._name_map_from_items(items)
            # Ensure deterministic ordering by id
            return dict(sorted(result.items()))
        except Exception as e:
            logger.exception("Failed to fetch document types: %s", e)
            return {}

    async def document_types_pretty(self, client: Optional[httpx.AsyncClient] = None) -> str:
        """
        Return a pretty-printed string of document type id -> name pairs (one per line).
        """
        return self._pretty_from_map(await self.document_types_name_map(client=client))

    async def correspondents_name_map(self, client: Optional[httpx.AsyncClient] = None) -> Dict[int, str]:
        """
        Return a simplified mapping of correspondent id -> correspondent name.
        """
        try:
            items = await self.list_correspondents(client=client)
            result = self._name_map_from_items(items)
            # Ensure deterministic ordering by id
            return dict(sorted(result.items()))
        except Exception as e:
            logger.exception("Failed to fetch correspondents: %s", e)
            return {}

    async def correspondents_pretty(self, client: Optional[httpx.AsyncClient] = None) -> str:
        """
        Return a pretty-printed string of correspondent id -> name pairs (one per line).
        """
        return self._pretty_from_map(await self.correspondents_name_map(client=client))


    async def create_tag(self, name: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        """
        Create a tag in the Paperless instance. Returns the created tag object on success,
        or an empty dict on failure. Adds additional debug logging on non-2xx responses.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        try:
            url = f"{self.base_url}/api/tags/"
            logger.info(f"Creating tag '{name}' via {url}")
            resp = await client.post(url, json={"name": name})
            # Prefer explicit handling so we can log body on failure
            try:
                resp.raise_for_status()
            except Exception:
                body = ""
                try:
                    body = resp.text
                except Exception:
                    body = "<unavailable>"
                logger.warning(f"create_tag failed: status={resp.status_code} body={body}")
                return {}

            try:
                return resp.json()
            except Exception:
                logger.warning("create_tag: response is not valid JSON")
                return {}
        except Exception as e:
            logger.exception("Failed to create tag: %s", e)
            return {}
        finally:
            if created_client:
                await created_client.aclose()


    async def create_document_type(self, name: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        """
        Create a document type in the Paperless instance. Returns the created document type object
        on success or an empty dict on failure. Adds additional debug logging on non-2xx responses.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        try:
            url = f"{self.base_url}/api/document_types/"
            logger.info(f"Creating document type '{name}' via {url}")
            resp = await client.post(url, json={"name": name})
            try:
                resp.raise_for_status()
            except Exception:
                body = ""
                try:
                    body = resp.text
                except Exception:
                    body = "<unavailable>"
                logger.warning(f"create_document_type failed: status={resp.status_code} body={body}")
                return {}

            try:
                return resp.json()
            except Exception:
                logger.warning("create_document_type: response is not valid JSON")
                return {}
        except Exception as e:
            logger.exception("Failed to create document type: %s", e)
            return {}
        finally:
            if created_client:
                await created_client.aclose()


    async def create_correspondent(
        self,
        name: str,
        email: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Create a correspondent/contact in the Paperless instance. Email is optional.
        Returns the created correspondent object on success, or an empty dict on failure.
        Adds additional debug logging on non-2xx responses.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        try:
            url = f"{self.base_url}/api/correspondents/"
            payload: Dict[str, Any] = {"name": name}
            if email:
                payload["email"] = email
            logger.info(f"Creating correspondent '{name}' via {url}")
            resp = await client.post(url, json=payload)
            try:
                resp.raise_for_status()
            except Exception:
                body = ""
                try:
                    body = resp.text
                except Exception:
                    body = "<unavailable>"
                logger.warning(f"create_correspondent failed: status={resp.status_code} body={body}")
                return {}

            try:
                return resp.json()
            except Exception:
                logger.warning("create_correspondent: response is not valid JSON")
                return {}
        except Exception as e:
            logger.exception("Failed to create correspondent: %s", e)
            return {}
        finally:
            if created_client:
                await created_client.aclose()


    async def get_or_create_correspondent(self, name: Optional[str], client: Optional[httpx.AsyncClient] = None) -> Optional[int]:
        """
        Return the correspondent id for the given name (case-insensitive). If no correspondent exists,
        attempt to create one and return its id. Returns None on failure or if name is empty.
        """
        if not name:
            return None
        try:
            # try to find existing correspondent by name (case-insensitive)
            name_map = await self.correspondents_name_map(client=client)
            target = name.strip().casefold()
            for k, v in name_map.items():
                if v and v.strip().casefold() == target:
                    return k
            # create if not found
            created = await self.create_correspondent(name, client=client)
            if created:
                # accept either 'id' or 'pk' in created object
                id_val = created.get("id") or created.get("pk")
                if id_val is not None:
                    return int(id_val)
        except Exception as e:
            logger.warning(f"get_or_create_correspondent failed for '{name}': {e}")
        return None

    async def get_or_create_tag(self, name: Optional[str], client: Optional[httpx.AsyncClient] = None) -> Optional[int]:
        """
        Return the tag id for the given name (case-insensitive). If no tag exists,
        attempt to create one and return its id. Returns None on failure or if name is empty.
        """
        if not name:
            return None
        try:
            name_map = await self.tags_name_map(client=client)
            target = name.strip().casefold()
            for k, v in name_map.items():
                if v and v.strip().casefold() == target:
                    return k
            created = await self.create_tag(name, client=client)
            if created:
                id_val = created.get("id") or created.get("pk")
                if id_val is not None:
                    return int(id_val)
        except Exception as e:
            logger.warning(f"get_or_create_tag failed for '{name}': {e}")
        return None

    async def get_or_create_document_type(self, name: Optional[str], client: Optional[httpx.AsyncClient] = None) -> Optional[int]:
        """
        Return the document type id for the given name (case-insensitive). If no document type exists,
        attempt to create one and return its id. Returns None on failure or if name is empty.
        """
        if not name:
            return None
        try:
            name_map = await self.document_types_name_map(client=client)
            target = name.strip().casefold()
            for k, v in name_map.items():
                if v and v.strip().casefold() == target:
                    return k
            created = await self.create_document_type(name, client=client)
            if created:
                id_val = created.get("id") or created.get("pk")
                if id_val is not None:
                    return int(id_val)
        except Exception as e:
            logger.warning(f"get_or_create_document_type failed for '{name}': {e}")
        return None

    async def update_document_from_metadata(
        self,
        doc_id: int,
        metadata: DocumentMetadata,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper to construct a Paperless document payload from a DocumentMetadata
        instance and PATCH the document. Ensures correspondent/tag/document_type exist (creating them if necessary).
        Returns the updated document object, or an empty dict on failure / no-op.
        """
        if not metadata:
            return {}

        try:
            payload: Dict[str, Any] = {}

            if metadata.title:
                payload["title"] = metadata.title

            # correspondent
            if metadata.correspondent:
                corr_id = await self.get_or_create_correspondent(metadata.correspondent, client=client)
                if corr_id is not None:
                    payload["correspondent"] = corr_id

            # tags
            tag_ids: List[int] = []
            for tag_name in metadata.tags or []:
                if not tag_name:
                    continue
                tid = await self.get_or_create_tag(tag_name, client=client)
                if tid is not None:
                    tag_ids.append(tid)
            if tag_ids:
                payload["tags"] = tag_ids

            # document type
            if metadata.document_type:
                dtid = await self.get_or_create_document_type(metadata.document_type, client=client)
                if dtid is not None:
                    payload["document_type"] = dtid

            # document date
            if metadata.document_date:
                try:
                    payload["document_date"] = metadata.document_date.isoformat()  # type: ignore[attr-defined]
                except Exception:
                    payload["document_date"] = str(metadata.document_date)

            if not payload:
                logger.info(f"No updatable fields found in metadata for document id={doc_id}")
                return {}

            updated = await self.update_document(doc_id, payload, client=client)
            return updated or {}
        except Exception as e:
            logger.exception(f"Failed to update document id={doc_id} from metadata: {e}")
            return {}


    async def update_document(
        self,
        doc_id: int,
        data: Dict[str, Any],
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing document by ID using PATCH with the provided data dict.
        Returns the updated document object on success, or an empty dict on failure.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        try:
            url = f"{self.base_url}/api/documents/{doc_id}/"
            logger.info(f"Patching document {doc_id} via {url} with data keys: {list(data.keys())}")
            resp = await client.patch(url, json=data)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.exception("Failed to update document %s: %s", doc_id, e)
            return {}
        finally:
            if created_client:
                await created_client.aclose()



if __name__ == "__main__":
    from dotenv import load_dotenv
    import yaml
    import json
    from loguru import logger
    from typing import Optional, Dict, Any
    from httpx import AsyncClient
    import sys
    import asyncio
    # Load environment variables from a local .env file (if present)
    load_dotenv()


    # Configuration via ENV (kept same names for backward compatibility)
    PAPERLESS_BASE_URL = os.getenv("PAPERLESS_BASE_URL")
    PAPERLESS_API_TOKEN = os.getenv("PAPERLESS_API_TOKEN")

    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

    # Use loguru for structured logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
    )

    if not PAPERLESS_BASE_URL:
        logger.error("Please set PAPERLESS_BASE_URL environment variable.")
        sys.exit(1)
    if not PAPERLESS_API_TOKEN:
        logger.error("Please set PAPERLESS_API_TOKEN environment variable.")
        sys.exit(1)
 

    client = PaperlessClient(PAPERLESS_BASE_URL,PAPERLESS_API_TOKEN)
    
    docs = asyncio.run(client.list_documents(limit=2))
    logger.info(f"{docs}")
