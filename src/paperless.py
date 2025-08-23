import os
from typing import Optional, List, Dict, Any

import httpx
from loguru import logger

class PaperlessClient:
    """
    Encapsulates communication with a paperless-ngx instance.

    Usage:
        client = PaperlessClient(base_url="https://paperless.example", api_token="TOKEN")
        async with httpx.AsyncClient(headers=client.headers, timeout=client.timeout) as session:
            docs = await client.list_documents(session, page_size=100, limit=10)
            data = await client.fetch_document_bytes(session, doc_id)
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

    async def list_documents(
        self,
        client: Optional[httpx.AsyncClient] = None,
        page_size: int = 100,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of document objects. Accepts an existing httpx.AsyncClient or
        will create/close its own client if 'client' is None.
        The method is tolerant of different shapes (DRF-style 'results'/'next' or direct lists).
        """
        created_client = None
        if client is None:
            created_client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout)
            client = created_client

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
        Accepts an httpx.AsyncClient instance (recommended).
        Returns bytes (image/pdf) or None.
        """
        if client is None:
            # Create a temporary client if not provided
            client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout)
            created_client = client
        else:
            created_client = None

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
        if client is None:
            created_client = httpx.AsyncClient(headers=self.headers, timeout=self.timeout)
            client = created_client

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
        endpoints = ["correspondents", "correspondent", "contacts"]
        return await self._fetch_list(client, endpoints, page_size=page_size, limit=limit)
