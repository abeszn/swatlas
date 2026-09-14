"""Typed configuration loaded from config.yaml plus credentials from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TIMEFRAMES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 16385, "H4": 16388, "D1": 16408,
}
"""MT5 timeframe constants, mirrored so config parsing does not need the terminal."""


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "ma_cross"          # ma_cross | donchian | smc
    fast_period: int = 20
    slow_period: int = 50
    entry_buffer_atr: float = 0.10
    params: dict = field(default_factory=dict)   # extra args for the chosen strategy
    min_history: int = 200          # bars of warm-up beyond the indicator windows


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float
    atr_period: int
    stop_loss_atr_mult: float
    take_profit_atr_mult: float
    max_open_positions: int
    max_daily_loss_pct: float
    max_lot: float
    min_lot: float = 0.0        # floor: size up to this even if risk maths says less
    fixed_lot: float = 0.0      # >0 trades this exact size, ignoring risk_per_trade_pct


@dataclass(frozen=True)
class NewsConfig:
    enabled: bool = True
    min_impact: str = "high"
    before_minutes: int = 30
    after_minutes: int = 45
    max_cache_age_hours: int = 12


@dataclass(frozen=True)
class ExecutionConfig:
    dry_run: bool
    allow_live_account: bool
    poll_seconds: int
    deviation_points: int


@dataclass(frozen=True)
class Credentials:
    login: int | None
    password: str | None
    server: str | None
    path: str | None


@dataclass(frozen=True)
class Config:
    symbol: str
    timeframe_name: str
    timeframe: int
    magic: int
    strategy: StrategyConfig
    risk: RiskConfig
    execution: ExecutionConfig
    news: NewsConfig
    credentials: Credentials
    broker_suffix: str = ""     # e.g. "m" on brokers that suffix standard-account
                                # symbols (XAUUSD -> XAUUSDm). Already applied to
                                # `symbol` above; scanner.py applies it to the
                                # basket, since DEFAULT_BASKET stays broker-neutral.

    @property
    def bars_needed(self) -> int:
        """Enough history for the slowest indicator plus slack for warm-up.

        Structure-based strategies need far more than their nominal windows -
        SMC has to see prior swings and order blocks form - hence min_history.
        """
        return max(self.strategy.slow_period, self.risk.atr_period,
                   self.strategy.min_history) + 100


def load_config(config_path: Path | str | None = None) -> Config:
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    tf_name = str(raw["timeframe"]).upper()
    if tf_name not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {sorted(TIMEFRAMES)}, got {tf_name!r}")

    strategy = StrategyConfig(**raw["strategy"])
    if strategy.fast_period >= strategy.slow_period:
        raise ValueError("strategy.fast_period must be less than strategy.slow_period")

    risk = RiskConfig(**raw["risk"])
    if not 0 < risk.risk_per_trade_pct <= 5:
        raise ValueError("risk.risk_per_trade_pct must be in (0, 5]; higher is not a bug budget")
    if risk.max_lot <= 0:
        raise ValueError("risk.max_lot must be positive")
    if risk.min_lot < 0:
        raise ValueError("risk.min_lot must not be negative")
    if risk.min_lot > risk.max_lot:
        raise ValueError("risk.min_lot must not exceed risk.max_lot")
    if risk.fixed_lot < 0:
        raise ValueError("risk.fixed_lot must not be negative")
    if risk.fixed_lot > risk.max_lot:
        raise ValueError("risk.fixed_lot must not exceed risk.max_lot")

    # Some brokers (e.g. Exness standard/demo accounts) suffix every symbol -
    # XAUUSD becomes XAUUSDm. config.yaml stays broker-neutral; the suffix is
    # appended once here rather than baked into `symbol`, so switching accounts
    # is a one-line change instead of rewriting every symbol reference.
    broker_suffix = str(raw.get("broker_suffix", ""))
    symbol = str(raw["symbol"])
    if broker_suffix and not symbol.endswith(broker_suffix):
        symbol += broker_suffix

    load_dotenv(PROJECT_ROOT / ".env")
    login = os.getenv("MT5_LOGIN")
    credentials = Credentials(
        login=int(login) if login else None,
        password=os.getenv("MT5_PASSWORD") or None,
        server=os.getenv("MT5_SERVER") or None,
        path=os.getenv("MT5_PATH") or None,
    )

    return Config(
        symbol=symbol,
        timeframe_name=tf_name,
        timeframe=TIMEFRAMES[tf_name],
        magic=int(raw["magic"]),
        strategy=strategy,
        risk=risk,
        broker_suffix=broker_suffix,
        execution=ExecutionConfig(**raw["execution"]),
        news=NewsConfig(**raw.get("news", {})),
        credentials=credentials,
    )
