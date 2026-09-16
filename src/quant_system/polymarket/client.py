"""Read-only HTTP access to Polymarket's public Gamma and CLOB APIs.

The client never signs, funds, or submits anything: it only reads public market
metadata and order books. Network access is injected through a small transport
protocol so that every consumer -- tests, scanners, and the research study --
can run against recorded snapshots without touching the network.

Snapshots are first-class rather than a debugging aid. Prediction-market books
are shallow and move quickly, so a reproducible result requires the exact
payload the decision was made from; :class:`SnapshotStore` persists it next to
the experiment output.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .markets import Market, parse_markets
from .orderbook import OrderBook

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


class PolymarketHTTPError(RuntimeError):
    """A response the venue rejected outright, as opposed to a transient failure.

    Kept as its own type so the retry loop can tell "your request was wrong"
    from "the network hiccuped" by class rather than by matching the error text.
    A string comparison there would turn every 4xx into four silent retries the
    first time someone reworded the message.
    """

    def __init__(self, method: str, url: str, status_code: int, body: str = "") -> None:
        super().__init__(f"{method} {url} failed with HTTP {status_code}: {body[:200]}")
        self.status_code = status_code


class HttpTransport(Protocol):
    """Minimal JSON transport, implemented by tests as an in-memory fake."""

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any: ...

    def post_json(self, url: str, body: Any) -> Any: ...


@dataclass
class RequestsTransport:
    """``requests``-backed transport with bounded retries on transient errors.

    Only connection failures, timeouts, and 5xx/429 responses are retried; a
    4xx response is a request problem and is raised immediately.
    """

    timeout: float = 20.0
    max_retries: int = 4
    backoff: float = 1.5
    user_agent: str = "ai-quant-trading-system/polymarket-research"

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        return self._request("GET", url, params=params)

    def post_json(self, url: str, body: Any) -> Any:
        return self._request("POST", url, json_body=body)

    def _request(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        import requests

        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code < 400:
                    return response.json()
                if response.status_code != 429 and response.status_code < 500:
                    raise PolymarketHTTPError(
                        method, url, response.status_code, response.text
                    )
                last_error = RuntimeError(f"HTTP {response.status_code} from {url}")
            except PolymarketHTTPError:
                raise  # the request itself was rejected; retrying cannot fix it
            except Exception as error:  # noqa: BLE001 - transient, retried below
                last_error = error
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff**attempt)
        raise RuntimeError(f"{method} {url} failed after {self.max_retries} attempts") from last_error


@dataclass
class PolymarketClient:
    """Public data access for markets, books, and price history."""

    transport: HttpTransport | None = None
    gamma_url: str = GAMMA_URL
    clob_url: str = CLOB_URL
    page_size: int = 100

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = RequestsTransport()

    def list_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        order: str = "volumeNum",
        ascending: bool = False,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """One page of raw Gamma market records."""
        params: dict[str, Any] = {
            "limit": int(limit),
            "offset": int(offset),
            "active": _bool_param(active),
            "closed": _bool_param(closed),
            "order": order,
            "ascending": _bool_param(ascending),
        }
        params.update({key: _bool_param(value) for key, value in filters.items()})
        payload = self.transport.get_json(f"{self.gamma_url}/markets", params)
        return _as_records(payload)

    def iter_markets(
        self,
        max_markets: int = 500,
        **filters: Any,
    ) -> Iterator[dict[str, Any]]:
        """Page through Gamma until ``max_markets`` or the listing runs out."""
        fetched = 0
        offset = 0
        while fetched < max_markets:
            page = self.list_markets(
                limit=min(self.page_size, max_markets - fetched),
                offset=offset,
                **filters,
            )
            if not page:
                return
            for record in page:
                yield record
                fetched += 1
            offset += len(page)

    def fetch_markets(self, max_markets: int = 500, **filters: Any) -> list[Market]:
        """Normalized markets, ready for scanning or feature construction."""
        return parse_markets(self.iter_markets(max_markets=max_markets, **filters))

    def order_book(self, token_id: str) -> OrderBook:
        """Live book for one outcome token."""
        payload = self.transport.get_json(f"{self.clob_url}/book", {"token_id": str(token_id)})
        return parse_book(payload, fallback_token_id=str(token_id))

    def order_books(self, token_ids: Sequence[str]) -> dict[str, OrderBook]:
        """Books for every requested token, batching where the venue allows it.

        A partial batch response is completed with individual requests rather
        than returned as-is. Silently handing back a short dict would look like
        a successful fetch while the risk gates quietly rejected every market
        whose book went missing, shrinking the tradable universe with no signal
        that anything had gone wrong.
        """
        tokens = [str(token) for token in token_ids]
        if not tokens:
            return {}

        books: dict[str, OrderBook] = {}
        try:
            payload = self.transport.post_json(
                f"{self.clob_url}/books",
                [{"token_id": token} for token in tokens],
            )
            wanted = set(tokens)
            for record in _as_records(payload):
                book = parse_book(record)
                if book.token_id in wanted:
                    books[book.token_id] = book
        except Exception:  # noqa: BLE001 - the batch endpoint is an optimization only
            books = {}

        for token in tokens:
            if token not in books:
                books[token] = self.order_book(token)
        return books

    def price_history(
        self,
        token_id: str,
        interval: str = "1d",
        fidelity: int | None = None,
    ) -> list[dict[str, float]]:
        """Time series of traded prices as ``{"t": epoch, "p": price}`` points."""
        params: dict[str, Any] = {"market": str(token_id), "interval": interval}
        if fidelity is not None:
            params["fidelity"] = int(fidelity)
        payload = self.transport.get_json(f"{self.clob_url}/prices-history", params)
        history = payload.get("history", []) if isinstance(payload, Mapping) else payload
        points = []
        for item in _as_records(history):
            timestamp, price = item.get("t"), item.get("p")
            if timestamp is None or price is None:
                continue
            points.append({"t": float(timestamp), "p": float(price)})
        return points


def parse_book(payload: Any, fallback_token_id: str = "") -> OrderBook:
    """Convert a CLOB ``/book`` payload into an :class:`OrderBook`."""
    if not isinstance(payload, Mapping):
        return OrderBook(token_id=fallback_token_id)
    token_id = str(payload.get("asset_id") or payload.get("token_id") or fallback_token_id)
    timestamp = payload.get("timestamp")
    return OrderBook.from_levels(
        token_id=token_id,
        bids=_levels(payload.get("bids")),
        asks=_levels(payload.get("asks")),
        timestamp=float(timestamp) if timestamp is not None else None,
    )


@dataclass
class SnapshotStore:
    """On-disk JSON snapshots keyed by name and capture time."""

    directory: str | Path = "data/polymarket"

    def path_for(self, name: str, stamp: str | None = None) -> Path:
        suffix = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return Path(self.directory) / f"{name}_{suffix}.json"

    def save(self, name: str, payload: Any, stamp: str | None = None) -> Path:
        """Write a snapshot and return its path."""
        path = self.path_for(name, stamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "name": name,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return path

    def latest(self, name: str) -> Path | None:
        """Most recent snapshot for ``name``, by filename ordering."""
        matches = sorted(Path(self.directory).glob(f"{name}_*.json"))
        return matches[-1] if matches else None

    def load(self, name: str) -> Any:
        """Payload of the most recent snapshot for ``name``."""
        return self._record(name).get("payload")

    def captured_at(self, name: str) -> datetime | None:
        """When the most recent snapshot was taken.

        A replay has to be evaluated at the snapshot's own clock, not at wall
        time: gates like "resolves too soon" would otherwise reject markets that
        were perfectly tradable at the moment the data was captured.
        """
        stamp = self._record(name).get("captured_utc")
        if not isinstance(stamp, str):
            return None
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    def _record(self, name: str) -> dict[str, Any]:
        path = self.latest(name)
        if path is None:
            raise FileNotFoundError(
                f"No snapshot named {name!r} under {self.directory}; "
                "run scripts/fetch_polymarket_data.py first"
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        return record if isinstance(record, Mapping) else {"payload": record}


def _levels(raw: Any) -> list[tuple[float, float]]:
    levels: list[tuple[float, float]] = []
    for item in _as_records(raw):
        try:
            price, size = float(item["price"]), float(item["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0.0 < price < 1.0 and size > 0:
            levels.append((price, size))
    return levels


def _as_records(payload: Any) -> list[dict[str, Any]]:
    """Normalize the several shapes these endpoints use for collections."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("data", "history", "markets", "results"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, Mapping)]
    return []


def _bool_param(value: Any) -> Any:
    """Gamma expects lowercase string booleans in the query string."""
    return str(value).lower() if isinstance(value, bool) else value


def build_market_url(slug: str) -> str:
    """Human-facing market URL, useful in reports and order plans."""
    return f"https://polymarket.com/market/{urllib.parse.quote(str(slug))}"


def token_ids(markets: Iterable[Market]) -> list[str]:
    """Every outcome token across ``markets``, de-duplicated and ordered."""
    seen: dict[str, None] = {}
    for market in markets:
        for outcome in market.outcomes:
            seen.setdefault(outcome.token_id, None)
    return list(seen)
