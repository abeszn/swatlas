"""Economic calendar filter.

Source: the ForexFactory weekly calendar feed at nfs.faireconomy.media, which
publishes the same data as forexfactory.com/calendar in JSON/XML/CSV/ICS.

Two things to understand before relying on this:

1. RATE LIMIT. The feed is limited across all four format URLs together, and
   the publisher's guidance is to download once per week and reuse it. This
   module caches to disk and refuses to refetch inside `min_refresh_hours`.

2. IT CANNOT BE BACKTESTED HERE. The feed serves the current week only, so
   there is no 2004-2026 calendar history to replay. That is less damaging than
   it sounds: a news filter is a COST control, not an alpha source. What it
   avoids is spread blowout, slippage, and gap-through-stop during releases -
   and the backtester models a CONSTANT spread, so it is structurally blind to
   exactly the harm this prevents. Backtesting it would mostly show "fewer
   trades, slightly less edge captured" while hiding the benefit. Judge it live,
   on execution quality, not on a historical equity curve.

If you later obtain historical calendar data, `NewsCalendar.from_events` takes
any list of events and the blackout logic works unchanged.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import PROJECT_ROOT

log = logging.getLogger("news")

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FALLBACK_URL = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_PATH = PROJECT_ROOT / "research" / "data" / "calendar_cache.json"

IMPACT_RANK = {"holiday": 0, "low": 1, "medium": 2, "high": 3}

# Metals are priced in USD and react to USD macro; there is no "XAU calendar".
METAL_PREFIXES = ("XAU", "XAG", "XPT", "XPD")


@dataclass(frozen=True)
class Event:
    when: datetime          # UTC
    currency: str
    title: str
    impact: str             # "high" | "medium" | "low" | "holiday"

    @property
    def rank(self) -> int:
        return IMPACT_RANK.get(self.impact.lower(), 0)


def currencies_for(symbol: str) -> set[str]:
    """Which currencies' news can move this symbol.

    EURJPY -> {EUR, JPY}. XAUUSD -> {USD}: gold's calendar exposure is the
    dollar side, there is no metal-specific release.
    """
    name = symbol.upper()
    if name.startswith(METAL_PREFIXES):
        quote = name[3:6] if len(name) >= 6 else "USD"
        return {quote or "USD"}
    if len(name) >= 6:
        return {name[:3], name[3:6]}
    return set()


class NewsCalendar:
    """Economic events plus the blackout logic built on them."""

    def __init__(self, events: list[Event]):
        self.events = sorted(events, key=lambda e: e.when)

    # ------------------------------------------------------------- loading

    @classmethod
    def from_events(cls, events: list[Event]) -> "NewsCalendar":
        return cls(events)

    @classmethod
    def load(cls, min_refresh_hours: int = 24, offline_ok: bool = True
             ) -> "NewsCalendar":
        """Cached fetch. Falls back to a stale cache rather than failing a
        trading decision on a network hiccup."""
        cached = cls._read_cache()
        if cached is not None:
            age_hours, events = cached
            if age_hours < min_refresh_hours:
                log.debug("Using calendar cache (%.1fh old, %d events)",
                          age_hours, len(events))
                return cls(events)

        try:
            events = cls._fetch()
            cls._write_cache(events)
            log.info("Fetched %d calendar events", len(events))
            return cls(events)
        except Exception as exc:
            if cached is not None:
                age_hours, events = cached
                log.warning("Calendar refresh failed (%s); using cache %.1fh old.",
                            exc, age_hours)
                return cls(events)
            if offline_ok:
                log.warning("Calendar unavailable (%s) and no cache. "
                            "Proceeding with NO news filter.", exc)
                return cls([])
            raise

    @staticmethod
    def _fetch() -> list[Event]:
        last_error: Exception | None = None
        for url in (FEED_URL, FALLBACK_URL):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "swatlas/0.1 (personal trading bot)"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                return _parse(raw)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                continue
        raise RuntimeError(f"calendar feed unreachable: {last_error}")

    @staticmethod
    def _read_cache() -> tuple[float, list[Event]] | None:
        if not CACHE_PATH.exists():
            return None
        try:
            payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(payload["fetched_at"])
            age = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
            events = [
                Event(datetime.fromisoformat(e["when"]), e["currency"],
                      e["title"], e["impact"])
                for e in payload["events"]
            ]
            return age, events
        except Exception as exc:
            log.debug("Calendar cache unreadable: %s", exc)
            return None

    @staticmethod
    def _write_cache(events: list[Event]) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps({
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "events": [
                {"when": e.when.isoformat(), "currency": e.currency,
                 "title": e.title, "impact": e.impact}
                for e in events
            ],
        }, indent=1), encoding="utf-8")

    # ------------------------------------------------------------ blackout

    def upcoming(self, symbol: str, now: datetime, within_minutes: int = 60,
                 min_impact: str = "high") -> list[Event]:
        floor = IMPACT_RANK.get(min_impact.lower(), 3)
        wanted = currencies_for(symbol)
        horizon = now + timedelta(minutes=within_minutes)
        return [e for e in self.events
                if e.rank >= floor and e.currency.upper() in wanted
                and now <= e.when <= horizon]

    def blackout(self, symbol: str, now: datetime, before_minutes: int = 30,
                 after_minutes: int = 30, min_impact: str = "high"
                 ) -> Event | None:
        """The event making `symbol` untradeable right now, or None if clear.

        Asymmetric windows are supported because the hazards differ: before a
        release spreads widen and liquidity thins; after it, price gaps and
        stops slip. `after` is often worth setting longer than `before`.
        """
        floor = IMPACT_RANK.get(min_impact.lower(), 3)
        wanted = currencies_for(symbol)
        for event in self.events:
            if event.rank < floor or event.currency.upper() not in wanted:
                continue
            if (event.when - timedelta(minutes=before_minutes)
                    <= now
                    <= event.when + timedelta(minutes=after_minutes)):
                return event
        return None


def _parse(raw: list) -> list[Event]:
    events: list[Event] = []
    for row in raw:
        stamp = row.get("date")
        if not stamp:
            continue
        try:
            when = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        when = (when.replace(tzinfo=timezone.utc) if when.tzinfo is None
                else when.astimezone(timezone.utc))
        events.append(Event(
            when=when,
            currency=(row.get("country") or "").upper(),
            title=row.get("title") or "",
            impact=(row.get("impact") or "").lower(),
        ))
    return events
