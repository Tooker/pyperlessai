import os
from typing import Optional, List, Dict, Any, Tuple

import httpx
from loguru import logger

class PaperlessClient:
    """
    Encapsulates communication with a paperless-ngx instance.

    Usage (preferred for long-running/bulk operations):
        async with PaperlessClient(base_url="https://paperless.example", api_token="TOKEN") as client:
            docs = await client.list_documents(page_size=100, limit=10)
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

    async def list_documents(
        self,
        client: Optional[httpx.AsyncClient] = None,
        page_size: int = 100,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of document objects. Accepts an existing httpx.AsyncClient (for
        backwards compatibility) or will use the shared client when PaperlessClient
        is used as a context manager. If no client is available, a temporary client
        will be created and closed automatically.

        The method is tolerant of different shapes (DRF-style 'results'/'next' or direct lists).
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        docs: List[Dict[str, Any]] = []
        next_url = f"{self.base_url}/api/documents/?page_size={page_size}"

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
                    return docs[:limit]

                next_url = data.get("next")
        finally:
            if created_client:
                await created_client.aclose()

        return docs

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
        page_size: int = 500,
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
                next_url = f"{self.base_url}/api/{ep}/?page_size={page_size}"
                try:
                    while next_url:
                        next_url = next_url.replace("http://","https://")
                        logger.info(f"Fetching {ep} page: {next_url}")
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
        page_size: int = 100,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of tag objects from the paperless API.
        Tries common tag endpoint paths.
        """
        endpoints = ["tags"]
        return await self._fetch_list(client, endpoints, page_size=page_size, limit=limit)


    async def list_document_types(
        self,
        client: Optional[httpx.AsyncClient] = None,
        page_size: int = 100,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of document type objects. Tries several common endpoint name variants.
        """
        endpoints = ["document_types"]
        return await self._fetch_list(client, endpoints, page_size=page_size, limit=limit)


    async def list_correspondents(
        self,
        client: Optional[httpx.AsyncClient] = None,
        page_size: int = 100,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of correspondent/contact objects. Tries a few possible endpoint names.
        """
        endpoints = ["correspondents"]
        return await self._fetch_list(client, endpoints, page_size=page_size, limit=limit)


    # Utilities to produce simplified id -> name maps and pretty-printed representations
    def _name_map_from_items(self, items: List[Dict[str, Any]]) -> Dict[int, str]:
        """
        Build a mapping of item id -> item name from a list of dict-like items.
        Skips items that do not contain both 'id' and 'name'.
        """
        result: Dict[int, str] = {}
        for item in items:
            try:
                if "id" in item and "name" in item:
                    result[int(item["id"])] = item.get("name", "")
            except Exception:
                # Ignore malformed items
                continue
        return result

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
        or an empty dict on failure.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        try:
            url = f"{self.base_url}/api/tags/"
            logger.info(f"Creating tag '{name}' via {url}")
            resp = await client.post(url, json={"name": name})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.exception("Failed to create tag: %s", e)
            return {}
        finally:
            if created_client:
                await created_client.aclose()


    async def create_document_type(self, name: str, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
        """
        Create a document type in the Paperless instance. Returns the created document type object
        on success or an empty dict on failure.
        """
        created_client = None
        client, created = await self._acquire_client(client)
        if created:
            created_client = client

        try:
            url = f"{self.base_url}/api/document_types/"
            logger.info(f"Creating document type '{name}' via {url}")
            resp = await client.post(url, json={"name": name})
            resp.raise_for_status()
            return resp.json()
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
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.exception("Failed to create correspondent: %s", e)
            return {}
        finally:
            if created_client:
                await created_client.aclose()


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
