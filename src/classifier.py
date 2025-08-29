from __future__ import annotations

from typing import Optional, Dict, Any, List, Any as AnyType
import asyncio
import json
import os
from pathlib import Path

from loguru import logger

from paperless import PaperlessClient
from ai_client import AIClient
from schemas import DocumentMetadata


class PaperlessOpenAIClassifier:
    """
    Classifier that prefetches Paperless entities (tags, document types, correspondents)
    and sends PDFs to OpenAI with these entities as grounded context inside the prompt.

    This version uses a normal __init__ (no async factory). The actual fetching of
    Paperless entities happens automatically during initialization:

    - If __init__ is called inside a running asyncio event loop, prefetch runs as a background task.
      Call `await classifier.ensure_prefetched()` to wait for completion when needed.

    - If __init__ is called outside of a running event loop (synchronous context),
      initialization will block and run the prefetch using asyncio.run(...).

    Constructor arguments:
      paperless: PaperlessClient
      ai: AIClient
      base_prompt: Optional[str] -- override for the base prompt
      prefetch: bool -- whether to fetch lists during init (default True)

    Methods:
      ensure_prefetched() -> awaitable: wait for prefetch to finish (if running in background)
      build_prompt(extra_instructions=None) -> str
      analyze_pdf(pdf_bytes, extra_instructions=None) -> result from AIClient.send_pdf_bytes
      analyze_paperless_document(doc_id, extra_instructions=None) -> fetch & analyze
    """

    def __init__(
        self,
        ai: AIClient,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        request_timeout: int = 30,
        base_prompt: Optional[str] = None,
        prefetch: bool = True,
    ):
        """
        The classifier always creates its own PaperlessClient instance during init.
        - base_url/api_token can be provided; otherwise environment variables are used.
        """
        if ai is None:
            raise ValueError("AIClient instance (ai) is required")
        self.ai = ai

        burl = base_url or os.getenv("PAPERLESS_BASE_URL")
        token = api_token or os.getenv("PAPERLESS_API_TOKEN")
        if not burl or not token:
            raise ValueError("PAPERLESS_BASE_URL and PAPERLESS_API_TOKEN must be provided either as arguments or environment variables")
        self.paperless = PaperlessClient(base_url=burl, api_token=token, request_timeout=request_timeout)

        self.base_prompt = base_prompt or (
            "Analyze the PDF and extract: title, correspondent, tags (max 3), "
            "document_date (YYYY-MM-DD), document_type, language. Prefer values from the allowed lists."
        )

        # placeholders for maps (populated by _prefetch)
        self.tags_map: Dict[int, str] = {}
        self.document_types_map: Dict[int, str] = {}
        self.correspondents_map: Dict[int, str] = {}

        # background prefetch task (if started)
        self._prefetch_task: Optional[asyncio.Task] = None
        # Remember whether prefetch was requested. Actual prefetching will be
        # started when entering the classifier's async context so the PaperlessClient's
        # shared session can be used for best performance.
        self._prefetch_requested = bool(prefetch)

        # Name of the tag used to mark documents already processed by AI.
        # Configurable via environment variable PAPERLESS_AI_PROCESSED_TAG (fallback: "AI_Processed")
        self.ai_processed_tag_name: str = os.getenv("PAPERLESS_AI_PROCESSED_TAG", "AI_Processed")

    async def _prefetch(self) -> None:
        """
        Internal async method to fetch tags, document types and correspondents.
        Relies on PaperlessClient to manage httpx client lifecycle.
        """
        logger.info("Fetching tags, document types, and correspondents from Paperless...")
        try:
            # Delegate error handling to PaperlessClient; its name-map helpers
            # will return empty dicts and log failures. Keep a top-level catch
            # for any unexpected errors during prefetch.
            self.tags_map = await self.paperless.tags_name_map()
            self.document_types_map = await self.paperless.document_types_name_map()
            self.correspondents_map = await self.paperless.correspondents_name_map()


            logger.info(f"Prefetch complete: {len(self.tags_map)} tags, {len(self.document_types_map)} document types, {len(self.correspondents_map)} correspondents")
        except Exception as e:
            logger.exception(f"Unexpected error during prefetch: {e}")

    async def ensure_prefetched(self) -> None:
        """
        Await this if a background prefetch task was started and you need to ensure maps are populated.
        Safe to call even if prefetch was run synchronously (no-op).
        """
        if self._prefetch_task:
            await self._prefetch_task
            self._prefetch_task = None

    # Helper: extract tag IDs from a Paperless document representation (robust to various shapes)
    def _extract_tag_ids_from_doc(self, doc: Dict[str, Any]) -> List[int]:
        """
        Inspect common places where Paperless document encodes tags and return a list of tag IDs.
        Handles:
         - integer IDs
         - dicts with "id" or "name"
         - string names (mapped via self.tags_map)
        Returns a de-duplicated list of ints.
        """
        tags_field = doc.get("tags") or doc.get("tag_set") or doc.get("tag_ids") or []
        result: List[int] = []
        for t in tags_field:
            try:
                if isinstance(t, int):
                    result.append(int(t))
                elif isinstance(t, dict):
                    if "id" in t and t["id"] is not None:
                        result.append(int(t["id"]))
                    elif "name" in t and t["name"]:
                        name = str(t["name"]).strip().casefold()
                        for k, v in self.tags_map.items():
                            if v and v.strip().casefold() == name:
                                result.append(k)
                                break
                elif isinstance(t, str):
                    name = t.strip().casefold()
                    for k, v in self.tags_map.items():
                        if v and v.strip().casefold() == name:
                            result.append(k)
                            break
            except Exception:
                # ignore malformed entries
                continue

        # preserve order and deduplicate
        seen = set()
        uniq: List[int] = []
        for x in result:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _doc_has_ai_processed_tag(self, doc: Dict[str, Any]) -> bool:
        """
        Return True if the document already contains the AI_Processed tag.
        Checks both resolved tag IDs (using tags_map) and raw tag name entries.
        """
        ai_name = getattr(self, "ai_processed_tag_name", os.getenv("PAPERLESS_AI_PROCESSED_TAG", "AI_Processed"))
        ai_name_cf = ai_name.casefold()
        ai_id = None
        # Try to find the tag id in the prefetched tags_map
        for k, v in self.tags_map.items():
            if v and v.strip().casefold() == ai_name_cf:
                ai_id = k
                break

        tag_ids = self._extract_tag_ids_from_doc(doc)
        if ai_id is not None and ai_id in tag_ids:
            return True

        # Fallback: check raw tag name entries in the document
        for t in doc.get("tags") or []:
            try:
                if isinstance(t, dict) and "name" in t and str(t["name"]).strip().casefold() == ai_name.casefold():
                    return True
                if isinstance(t, str) and t.strip().casefold() == ai_name.casefold():
                    return True
            except Exception:
                continue

        return False

    async def _add_ai_processed_tag(self, doc_id: int, doc: Dict[str, Any]) -> None:
        """
        Ensure the AI_Processed tag exists and append it to the document's tags (without removing existing tags).
        Uses PaperlessClient.get_or_create_tag and update_document.
        """
        try:
            ai_tag_id = await self.paperless.get_or_create_tag(self.ai_processed_tag_name)
            if not ai_tag_id:
                logger.warning(f"Could not obtain id for {self.ai_processed_tag_name} tag")
                return

            current_ids = self._extract_tag_ids_from_doc(doc)
            if ai_tag_id in current_ids:
                # already present
                return

            new_ids = current_ids + [ai_tag_id]
            updated = await self.paperless.update_document(doc_id, {"tags": new_ids})
            if updated:
                logger.info(f"Added AI_Processed tag to document id={doc_id}")
            else:
                logger.warning(f"Failed to add AI_Processed tag to document id={doc_id}")
        except Exception as e:
            logger.warning(f"Exception while adding AI_Processed tag to doc {doc_id}: {e}")

    async def __aenter__(self) -> "PaperlessOpenAIClassifier":
        """
        Open the internal PaperlessClient shared session and start prefetch (non-blocking)
        if it was requested during initialization.
        """
        # Open shared PaperlessClient session
        await self.paperless.__aenter__()

        # Start prefetch after shared session is active for best performance.
        if self._prefetch_requested:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop:
                logger.info("Starting background prefetch task for Paperless entities (in __aenter__).")
                self._prefetch_task = loop.create_task(self._prefetch())
            else:
                logger.info("Running synchronous prefetch for Paperless entities (in __aenter__).")
                await self._prefetch()

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """
        Ensure any background prefetch finishes and close the internal PaperlessClient session.
        """
        try:
            await self.ensure_prefetched()
        finally:
            await self.paperless.__aexit__(exc_type, exc, tb)

    async def list_documents(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Convenience wrapper that delegates to the internal PaperlessClient.
        """
        return await self.paperless.list_documents(limit=limit)

    def _format_allowed_section(self, title: str, mapping: Dict[int, str]) -> str:
        lines = [f"{k}: {v}" for k, v in mapping.items()]
        return f"{title} (id -> name):\n" + ("\n".join(lines) if lines else "(none)")

    def build_prompt(self, extra_instructions: Optional[str] = None) -> str:
        """
        Build the final prompt by loading an external Markdown template
        (prompts/paperless_prompt.md) and substituting the placeholders:

        - {{known_correspondents}} -> JSON array of correspondent names
        - {{tags}} -> JSON array of tag names
        - {{document_types}} -> JSON array of document type names

        The template keeps the {{document_text}} placeholder so callers can insert
        the document content separately (or let the AI client handle it).
        """
        # Prepare allowed lists as arrays of names (preserve order)
        correspondents = list(self.correspondents_map.values())
        tags = list(self.tags_map.values())
        document_types = list(self.document_types_map.values())

        # Attempt to load the external template (relative to project root)
        try:
            template_path = Path(__file__).resolve().parents[1] / "prompts" / "paperless_prompt.md"
            template = template_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.exception(f"Failed to read prompt template, falling back to inlined prompt: {e}")
            # Minimal fallback if the external template cannot be read
            template = self.base_prompt

        # Render lists as JSON arrays (preserve unicode characters)
        try:
            known_correspondents_str = json.dumps(correspondents, ensure_ascii=False)
            tags_str = json.dumps(tags, ensure_ascii=False)
            document_types_str = json.dumps(document_types, ensure_ascii=False)
        except Exception:
            known_correspondents_str = str(correspondents)
            tags_str = str(tags)
            document_types_str = str(document_types)

        prompt_text = template.replace("{{known_correspondents}}", known_correspondents_str)
        prompt_text = prompt_text.replace("{{tags}}", tags_str)
        prompt_text = prompt_text.replace("{{document_types}}", document_types_str)

        if extra_instructions:
            prompt_text = "Extra instructions:\n" + extra_instructions + "\n\n" + prompt_text

        return prompt_text

    async def analyze_pdf(
        self,
        pdf_bytes: bytes,
        extra_instructions: Optional[str] = None,
        paperless_doc_id: Optional[int] = None,
    ) -> AnyType:
        """
        Send the provided PDF to OpenAI using AIClient, with a prompt that
        includes grounded context (tags, correspondents, document types).

        After receiving a structured response from the AI (DocumentMetadata),
        ensure Paperless contains the referenced correspondent, tags and document_type,
        creating any missing entities. If paperless_doc_id is provided, patch the
        Paperless document to set the correspondent, tags, document_type, title and document_date.

        Returns the raw response from AIClient.send_pdf_bytes (SDK object or dict).
        """
        # Refresh only tags so the grounded prompt uses the latest tag names
        try:
            self.tags_map = await self.paperless.tags_name_map()
        except Exception:
            logger.warning("Failed to refresh tags before sending PDF to OpenAI; proceeding with cached tags")
        prompt_text = self.build_prompt(extra_instructions=extra_instructions)
        logger.debug("Sending PDF to OpenAI with grounded context (tags, correspondents, document types).")
        result = await self.ai.send_pdf_bytes(pdf_bytes, prompt_text)

        # Attempt to extract parsed output that matches DocumentMetadata
        parsed_obj = None
        try:
            # Common SDK/dict locations
            if hasattr(result, "output_parsed"):
                parsed_obj = getattr(result, "output_parsed")
            elif isinstance(result, dict) and "output_parsed" in result:
                parsed_obj = result["output_parsed"]
        except Exception:
            parsed_obj = None

        # If we found something that looks like parsed output, try to coerce into the Pydantic model.
        metadata: Optional[DocumentMetadata] = None
        if parsed_obj is not None:
            try:
                # If parsed_obj is a JSON/string, try to load it
                if isinstance(parsed_obj, str):
                    try:
                        parsed_dict = json.loads(parsed_obj)
                    except Exception:
                        parsed_dict = None
                elif isinstance(parsed_obj, dict):
                    parsed_dict = parsed_obj
                else:
                    # Fallback: try to call model_dump or convert to dict
                    try:
                        parsed_dict = parsed_obj.model_dump()  # type: ignore[attr-defined]
                    except Exception:
                        parsed_dict = None

                if parsed_dict is not None:
                    metadata = DocumentMetadata.model_validate(parsed_dict)
            except Exception as e:
                logger.warning(f"Failed to parse AI output into DocumentMetadata: {e}")
                metadata = None

        # If we didn't get structured metadata, nothing more to do — return raw result.
        if metadata is None:
            logger.info("No structured metadata found in AI response; returning raw result.")
            return result

        # Ensure we have the latest name maps
        await self.ensure_prefetched()

        # Delegate creation/lookup and payload construction to PaperlessClient
        if paperless_doc_id is not None:
            try:
                updated = await self.paperless.update_document_from_metadata(paperless_doc_id, metadata)
                if updated:
                    logger.info(f"Updated Paperless document id={paperless_doc_id} from metadata")
                else:
                    logger.warning(f"Paperless update returned no result for id={paperless_doc_id}")
            except Exception as e:
                logger.warning(f"Failed to update Paperless document id={paperless_doc_id}: {e}")

        return result

    async def analyze_paperless_document(
        self,
        doc_id: int,
        extra_instructions: Optional[str] = None,
    ) -> AnyType:
        """
        Fetch a document from Paperless by id, then analyze it.
        - If it's a PDF, upload to OpenAI with grounded prompt.
        - Otherwise, treat content as image bytes and use vision endpoint.

        If a background prefetch is still running, this method will wait for it to finish
        to ensure grounded lists are available.
        """
        # Wait for prefetch to finish if it's in progress
        await self.ensure_prefetched()

        content = await self.paperless.fetch_document_bytes(None, doc_id)
        if not content:
            raise ValueError(f"No content returned for Paperless document id={doc_id}")

        if content.startswith(b"%PDF"):
            return await self.analyze_pdf(content, extra_instructions=extra_instructions, paperless_doc_id=doc_id)

        try:
            self.tags_map = await self.paperless.tags_name_map()
        except Exception:
            logger.warning("Failed to refresh tags before sending image to OpenAI; proceeding with cached tags")
        logger.info("Sending image bytes to AI vision endpoint.")
        return await self.ai.send_image_bytes(content, self.build_prompt(extra_instructions=extra_instructions))

    async def _process_document(
        self,
        semaphore: asyncio.Semaphore,
        doc: Dict[str, Any],
        out_dir: str = "poc_output",
    ) -> None:
        """
        Internal per-document processing previously located in poc.py.
        Fetches document bytes via self.paperless, analyzes with the classifier/AI client,
        and writes JSON output files under out_dir.
        """
        async with semaphore:
            doc_id = doc.get("id")
            title = doc.get("title") or "<untitled>"
            logger.info(f"Processing doc id={doc_id} title={title}")
            if not doc_id:
                logger.warning(f"No id found on document: {doc}")
                return

            # Ensure prefetched maps are available before checking tags
            await self.ensure_prefetched()
            # If document already processed (has AI_Processed tag), skip it
            if self._doc_has_ai_processed_tag(doc):
                logger.info(f"Skipping doc id={doc_id} because it already has AI_Processed tag")
                return

            os.makedirs(out_dir, exist_ok=True)

        logger.info(f"Analyzing Paperless document id={doc_id} with classifier")
        try:
            result = await self.analyze_paperless_document(doc_id, extra_instructions=None)

            # # Convert result to JSON for logging/saving in a robust way that handles SDK objects or dicts.
            # raw_json = ""
            # try:
            #     if hasattr(result, "model_dump_json"):
            #         raw_json = result.model_dump_json(indent=2)
            #     else:
            #         raw_json = json.dumps(result, ensure_ascii=False, indent=2)
            # except Exception:
            #     try:
            #         raw_json = json.dumps({"raw": str(result)}, ensure_ascii=False, indent=2)
            #     except Exception:
            #         raw_json = str(result)

            # json_out = os.path.join(out_dir, f"{doc_id}.openai.json")
            # json_parsed_out = os.path.join(out_dir, f"{doc_id}.openai_parsed.json")

            # with open(json_out, "w", encoding="utf-8") as jf:
            #     jf.write(raw_json)

            # # Attempt to save parsed output if available on the SDK object or dict
            # try:
            #     if hasattr(result, "output_parsed"):
            #         parsed_json = result.output_parsed.model_dump_json(indent=2)
            #         with open(json_parsed_out, "w", encoding="utf-8") as parsed:
            #             parsed.write(parsed_json)
            #     elif isinstance(result, dict) and "output_parsed" in result:
            #         with open(json_parsed_out, "w", encoding="utf-8") as parsed:
            #             parsed.write(json.dumps(result["output_parsed"], ensure_ascii=False, indent=2))
            # except Exception:
            #     # Ignore parsing save errors; main result is already saved.
            #     pass

            # After successful processing, attempt to add the AI_Processed tag to the Paperless document.
            try:
                await self._add_ai_processed_tag(doc_id, doc)
            except Exception:
                # _add_ai_processed_tag logs its own errors; don't let tagging failures interrupt the main flow.
                pass

        except Exception as e:
            logger.exception(f"Failed to call classifier for doc {doc_id}: {e}")
        return

    async def run(
        self,
        limit: int = 10,
        concurrent_workers: int = 4,
        max_image_size: int = 1024,
        out_dir: str = "poc_output",
    ) -> None:
        """
        Convenience method to run the end-to-end flow:
         - open internal PaperlessClient session
         - list documents
         - ensure prefetch
         - process documents concurrently
        """
        # Ensure output directory exists before running
        os.makedirs(out_dir, exist_ok=True)

        async with self:
            docs = await self.list_documents(limit=limit)
            logger.info(f"Found {len(docs)} documents (requested limit={limit})")

            # Wait for classifier prefetch to finish (if it ran in background)
            await self.ensure_prefetched()

            semaphore = asyncio.Semaphore(concurrent_workers)
            tasks = []
            for doc in docs:
                tasks.append(self._process_document(semaphore, doc, out_dir=out_dir))
            await asyncio.gather(*tasks)
