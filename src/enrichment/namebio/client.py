"""
NameBio microservice REST client.

Wraps the existing NameBio Spring Boot API (read-only) and maps its
camelCase `NamebioSale` JSON onto the snake_case shape the agent's
retrieval/scoring code expects:

    NameBio field   -> agent field
    -------------------------------
    domain          -> domain
    price           -> price
    saleDate        -> date      (ISO yyyy-mm-dd)
    marketplace     -> platform

Endpoints used:
    GET {base}/namebio/health
    GET {base}/namebio/sales?date=YYYY-MM-DD&page=&size=   (paginated)

Reliability: retry with exponential backoff (tenacity) on transient HTTP /
network errors; pages are fetched until the Spring Data `last` flag is true.
"""

import logging
from datetime import date
from typing import Dict, Iterator, List, Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

logger = logging.getLogger(__name__)


class NamebioClientError(Exception):
    """Raised when NameBio returns a non-retryable error."""


# Exceptions worth retrying (transient network / 5xx).
_RETRYABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class NamebioClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        page_size: Optional[int] = None,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = (base_url or config.NAMEBIO_BASE_URL).rstrip("/")
        self.page_size = page_size or config.NAMEBIO_PAGE_SIZE
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Low-level HTTP with retry
    # ------------------------------------------------------------------ #
    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        # Retry on 5xx by raising a retryable error; fail fast on 4xx.
        if resp.status_code >= 500:
            raise requests.exceptions.ConnectionError(
                f"NameBio {resp.status_code} for {url}"
            )
        if resp.status_code >= 400:
            raise NamebioClientError(
                f"NameBio {resp.status_code} for {url}: {resp.text[:200]}"
            )
        return resp.json()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def health(self) -> bool:
        """Return True if the microservice reports ok."""
        try:
            data = self._get("/namebio/health")
            return data.get("status") == "ok"
        except Exception as e:  # noqa: BLE001 - health check must not throw
            logger.warning("NameBio health check failed: %s", e)
            return False

    @staticmethod
    def _map_sale(raw: Dict) -> Optional[Dict]:
        """Map a raw NamebioSale JSON object to the agent's sale shape."""
        domain = raw.get("domain")
        if not domain:
            return None
        return {
            "domain": str(domain).strip().lower(),
            "price": float(raw["price"]) if raw.get("price") is not None else 0.0,
            "date": raw.get("saleDate"),          # already ISO yyyy-mm-dd
            "platform": raw.get("marketplace"),
        }

    def get_sales_for_date(self, day: date) -> List[Dict]:
        """
        Fetch ALL sales for a given date, paging through results.

        Returns a list of mapped sale dicts. A single bad page is allowed to
        raise; callers (ingest) isolate failures per date.
        """
        day_str = day.isoformat()
        sales: List[Dict] = []
        page = 0
        while True:
            payload = self._get(
                "/namebio/sales",
                params={"date": day_str, "page": page, "size": self.page_size},
            )
            content = payload.get("content", []) or []
            for raw in content:
                mapped = self._map_sale(raw)
                if mapped:
                    sales.append(mapped)

            # Spring Data Page: stop when `last` is true or page is empty.
            if payload.get("last", True) or not content:
                break
            page += 1
            # Safety guard against runaway paging.
            if page > 10000:
                logger.error("Runaway paging for %s, stopping at page %d", day_str, page)
                break

        logger.info("NameBio %s: fetched %d sales", day_str, len(sales))
        return sales

    def iter_sales_for_date(self, day: date) -> Iterator[Dict]:
        """Generator variant of get_sales_for_date (page-streaming)."""
        day_str = day.isoformat()
        page = 0
        while True:
            payload = self._get(
                "/namebio/sales",
                params={"date": day_str, "page": page, "size": self.page_size},
            )
            content = payload.get("content", []) or []
            for raw in content:
                mapped = self._map_sale(raw)
                if mapped:
                    yield mapped
            if payload.get("last", True) or not content:
                break
            page += 1
            if page > 10000:
                logger.error("Runaway paging for %s, stopping at page %d", day_str, page)
                break
