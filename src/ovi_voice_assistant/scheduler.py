"""Scheduler — persistent cron-based automations that run agent prompts proactively."""

import asyncio
import datetime
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# How often the scheduler checks for automations to fire.
_CHECK_INTERVAL_S = 30


@dataclass
class Automation:
    """A single scheduled automation."""

    id: str
    name: str
    schedule: str  # cron expression: minute hour dom month dow
    prompt: str  # what to ask the agent
    enabled: bool = True
    last_run: str = ""  # ISO timestamp — prevents double-firing within same minute


# ---------------------------------------------------------------------------
# Cron matching (no external dependency)
# ---------------------------------------------------------------------------


def _matches_field(field_expr: str, value: int) -> bool:
    """Check if a single cron field matches a value."""
    if field_expr == "*":
        return True
    for part in field_expr.split(","):
        if "/" in part:
            base, step_s = part.split("/", 1)
            step = int(step_s)
            start = 0 if base == "*" else int(base)
            if (value - start) >= 0 and (value - start) % step == 0:
                return True
        elif "-" in part:
            lo, hi = part.split("-", 1)
            if int(lo) <= value <= int(hi):
                return True
        else:
            if int(part) == value:
                return True
    return False


def cron_matches(expr: str, dt: datetime.datetime) -> bool:
    """Check if a 5-field cron expression matches a datetime.

    Fields: minute hour day-of-month month day-of-week
    Day-of-week: 0 = Sunday, 6 = Saturday (standard cron convention).
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        return False
    minute, hour, dom, month, dow = parts
    # Python isoweekday: 1=Mon..7=Sun → cron: 0=Sun, 1=Mon..6=Sat
    cron_dow = dt.isoweekday() % 7
    return (
        _matches_field(minute, dt.minute)
        and _matches_field(hour, dt.hour)
        and _matches_field(dom, dt.day)
        and _matches_field(month, dt.month)
        and _matches_field(dow, cron_dow)
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

# Callback types for the scheduler.
RunPrompt = Callable[[str], Awaitable[str]]
AnnounceAll = Callable[[str], Awaitable[None]]


class Scheduler:
    """Runs persistent cron automations in the background.

    Each automation has a cron schedule and a prompt.  When the schedule
    fires, the prompt is sent to the agent and the response is announced
    on all connected devices.
    """

    def __init__(
        self,
        path: Path,
        run_prompt: RunPrompt,
        announce: AnnounceAll,
    ) -> None:
        self._path = path
        self._run_prompt = run_prompt
        self._announce = announce
        self._automations: list[Automation] = []
        self._task: asyncio.Task | None = None
        self._fire_tasks: set[asyncio.Task] = set()

    # -- Persistence --

    def load(self) -> None:
        """Load automations from disk."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._automations = [Automation(**a) for a in data]
                logger.info(
                    "Loaded %d automation(s) from %s",
                    len(self._automations),
                    self._path,
                )
            except Exception:
                logger.exception("Failed to load automations from %s", self._path)
                self._automations = []
        else:
            self._automations = []

    def _save(self) -> None:
        """Persist automations to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([asdict(a) for a in self._automations], indent=2)
        )

    # -- Public API (called by agent tools) --

    @property
    def automations(self) -> list[Automation]:
        return list(self._automations)

    def create(self, name: str, schedule: str, prompt: str) -> Automation:
        """Create and persist a new automation."""
        # Validate cron expression
        parts = schedule.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: {schedule!r}. "
                "Expected 5 fields: minute hour day-of-month month day-of-week"
            )
        auto = Automation(
            id=uuid.uuid4().hex[:8],
            name=name,
            schedule=schedule,
            prompt=prompt,
        )
        self._automations.append(auto)
        self._save()
        logger.info("Automation created: %s (%s) → %r", name, schedule, prompt[:60])
        return auto

    def delete(self, name: str) -> bool:
        """Delete an automation by name. Returns True if found."""
        before = len(self._automations)
        self._automations = [a for a in self._automations if a.name != name]
        if len(self._automations) < before:
            self._save()
            logger.info("Automation deleted: %s", name)
            return True
        return False

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable an automation by name. Returns True if found."""
        for a in self._automations:
            if a.name == name:
                a.enabled = enabled
                self._save()
                logger.info(
                    "Automation %s: %s", "enabled" if enabled else "disabled", name
                )
                return True
        return False

    # -- Background loop --

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="scheduler")
        logger.info("Scheduler started (%d automations)", len(self._automations))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("Scheduler stopped")

    async def _loop(self) -> None:
        while True:
            try:
                await self._check()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler check error")
            await asyncio.sleep(_CHECK_INTERVAL_S)

    async def _check(self) -> None:
        now = datetime.datetime.now().astimezone()
        now_minute = now.replace(second=0, microsecond=0)

        for auto in self._automations:
            if not auto.enabled:
                continue
            if not cron_matches(auto.schedule, now):
                continue
            # Prevent double-firing within the same minute.
            if auto.last_run:
                last = datetime.datetime.fromisoformat(auto.last_run)
                if last.replace(second=0, microsecond=0) >= now_minute:
                    continue
            auto.last_run = now.isoformat()
            self._save()
            task = asyncio.create_task(
                self._fire(auto),
                name=f"automation-{auto.name}",
            )
            self._fire_tasks.add(task)
            task.add_done_callback(self._fire_tasks.discard)

    async def _fire(self, auto: Automation) -> None:
        """Run the automation's prompt through the agent and announce the result."""
        logger.info("Automation firing: %s → %r", auto.name, auto.prompt[:60])
        try:
            response = await self._run_prompt(auto.prompt)
            if response.strip():
                await self._announce(response)
                logger.info("Automation announced: %s → %r", auto.name, response[:80])
            else:
                logger.warning("Automation %s produced empty response", auto.name)
        except Exception:
            logger.exception("Automation %s failed", auto.name)
