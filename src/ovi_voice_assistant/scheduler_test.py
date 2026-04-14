"""Tests for the scheduler — cron matching, automation CRUD, and firing."""

import asyncio
import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ovi_voice_assistant.scheduler import (
    Automation,
    Scheduler,
    _matches_field,
    cron_matches,
)

# ---------------------------------------------------------------------------
# cron matching
# ---------------------------------------------------------------------------


class TestMatchesField:
    def test_star(self):
        assert _matches_field("*", 0)
        assert _matches_field("*", 59)

    def test_exact(self):
        assert _matches_field("5", 5)
        assert not _matches_field("5", 6)

    def test_range(self):
        assert _matches_field("1-5", 3)
        assert _matches_field("1-5", 1)
        assert _matches_field("1-5", 5)
        assert not _matches_field("1-5", 6)

    def test_step(self):
        assert _matches_field("*/15", 0)
        assert _matches_field("*/15", 15)
        assert _matches_field("*/15", 30)
        assert not _matches_field("*/15", 7)

    def test_step_with_base(self):
        assert _matches_field("5/10", 5)
        assert _matches_field("5/10", 15)
        assert _matches_field("5/10", 25)
        assert not _matches_field("5/10", 10)

    def test_list(self):
        assert _matches_field("1,3,5", 3)
        assert _matches_field("1,3,5", 5)
        assert not _matches_field("1,3,5", 2)

    def test_combined(self):
        assert _matches_field("1-3,10,*/20", 2)
        assert _matches_field("1-3,10,*/20", 10)
        assert _matches_field("1-3,10,*/20", 20)
        assert not _matches_field("1-3,10,*/20", 7)


class TestCronMatches:
    def test_every_minute(self):
        dt = datetime.datetime(2026, 4, 10, 12, 30)
        assert cron_matches("* * * * *", dt)

    def test_specific_time(self):
        dt = datetime.datetime(2026, 4, 10, 7, 0)
        assert cron_matches("0 7 * * *", dt)
        assert not cron_matches("0 8 * * *", dt)

    def test_weekday_filter(self):
        # 2026-04-10 is a Friday → isoweekday=5, cron dow=5
        dt = datetime.datetime(2026, 4, 10, 7, 0)
        assert cron_matches("0 7 * * 5", dt)  # Friday
        assert not cron_matches("0 7 * * 1", dt)  # Monday

    def test_weekday_range(self):
        # Monday through Friday = 1-5
        monday = datetime.datetime(2026, 4, 6, 9, 0)  # Monday
        saturday = datetime.datetime(2026, 4, 11, 9, 0)  # Saturday
        assert cron_matches("0 9 * * 1-5", monday)
        assert not cron_matches("0 9 * * 1-5", saturday)

    def test_sunday(self):
        # 2026-04-12 is a Sunday → isoweekday=7 → cron dow=0
        dt = datetime.datetime(2026, 4, 12, 10, 0)
        assert cron_matches("0 10 * * 0", dt)

    def test_every_30_minutes(self):
        assert cron_matches("*/30 * * * *", datetime.datetime(2026, 1, 1, 0, 0))
        assert cron_matches("*/30 * * * *", datetime.datetime(2026, 1, 1, 0, 30))
        assert not cron_matches("*/30 * * * *", datetime.datetime(2026, 1, 1, 0, 15))

    def test_invalid_expression(self):
        dt = datetime.datetime(2026, 1, 1, 0, 0)
        assert not cron_matches("bad", dt)
        assert not cron_matches("* * *", dt)

    def test_month_filter(self):
        jan = datetime.datetime(2026, 1, 1, 0, 0)
        feb = datetime.datetime(2026, 2, 1, 0, 0)
        assert cron_matches("0 0 1 1 *", jan)
        assert not cron_matches("0 0 1 1 *", feb)


# ---------------------------------------------------------------------------
# Scheduler CRUD
# ---------------------------------------------------------------------------


@pytest.fixture
def scheduler(tmp_path: Path):
    path = tmp_path / "automations.json"
    return Scheduler(
        path, run_prompt=AsyncMock(return_value="test response"), announce=AsyncMock()
    )


class TestSchedulerCRUD:
    def test_create_and_list(self, scheduler: Scheduler):
        auto = scheduler.create("morning", "0 7 * * *", "What's the weather?")
        assert auto.name == "morning"
        assert auto.schedule == "0 7 * * *"
        assert len(scheduler.automations) == 1

    def test_create_invalid_cron(self, scheduler: Scheduler):
        with pytest.raises(ValueError, match="Invalid cron"):
            scheduler.create("bad", "not a cron", "test")

    def test_delete(self, scheduler: Scheduler):
        scheduler.create("test", "0 7 * * *", "prompt")
        assert scheduler.delete("test")
        assert len(scheduler.automations) == 0

    def test_delete_not_found(self, scheduler: Scheduler):
        assert not scheduler.delete("nonexistent")

    def test_enable_disable(self, scheduler: Scheduler):
        scheduler.create("test", "0 7 * * *", "prompt")
        assert scheduler.set_enabled("test", False)
        assert not scheduler.automations[0].enabled
        assert scheduler.set_enabled("test", True)
        assert scheduler.automations[0].enabled

    def test_enable_not_found(self, scheduler: Scheduler):
        assert not scheduler.set_enabled("nonexistent", True)

    def test_persistence(self, tmp_path: Path):
        path = tmp_path / "automations.json"
        s1 = Scheduler(path, run_prompt=AsyncMock(), announce=AsyncMock())
        s1.create("morning", "0 7 * * *", "weather?")
        s1.create("evening", "0 18 * * *", "summary?")

        # New scheduler instance should load the saved automations
        s2 = Scheduler(path, run_prompt=AsyncMock(), announce=AsyncMock())
        s2.load()
        assert len(s2.automations) == 2
        assert s2.automations[0].name == "morning"
        assert s2.automations[1].name == "evening"

    def test_load_empty(self, scheduler: Scheduler):
        scheduler.load()
        assert len(scheduler.automations) == 0

    def test_load_corrupt_file(self, tmp_path: Path):
        path = tmp_path / "automations.json"
        path.write_text("not json!")
        s = Scheduler(path, run_prompt=AsyncMock(), announce=AsyncMock())
        s.load()
        assert len(s.automations) == 0


# ---------------------------------------------------------------------------
# Scheduler firing
# ---------------------------------------------------------------------------


class TestSchedulerFiring:
    async def test_fire_calls_prompt_and_announce(self, scheduler: Scheduler):
        auto = Automation(
            id="abc",
            name="test",
            schedule="* * * * *",  # every minute
            prompt="hello?",
        )
        await scheduler._fire(auto)
        scheduler._run_prompt.assert_awaited_once_with("hello?")
        scheduler._announce.assert_awaited_once_with("test response")

    async def test_fire_empty_response_does_not_announce(self, tmp_path: Path):
        path = tmp_path / "automations.json"
        announce = AsyncMock()
        s = Scheduler(path, run_prompt=AsyncMock(return_value=""), announce=announce)
        auto = Automation(id="abc", name="test", schedule="* * * * *", prompt="hello?")
        await s._fire(auto)
        announce.assert_not_awaited()

    async def test_check_fires_matching_automation(self, scheduler: Scheduler):
        scheduler.create("every-minute", "* * * * *", "test prompt")
        await scheduler._check()
        await asyncio.sleep(0)  # let the created task run
        scheduler._run_prompt.assert_awaited_once_with("test prompt")

    async def test_check_skips_disabled(self, scheduler: Scheduler):
        scheduler.create("disabled", "* * * * *", "test")
        scheduler.set_enabled("disabled", False)
        await scheduler._check()
        scheduler._run_prompt.assert_not_awaited()

    async def test_check_prevents_double_fire(self, scheduler: Scheduler):
        scheduler.create("every-minute", "* * * * *", "test")
        await scheduler._check()
        scheduler._run_prompt.reset_mock()
        # Second check in same minute should not fire again
        await scheduler._check()
        scheduler._run_prompt.assert_not_awaited()

    async def test_start_stop(self, scheduler: Scheduler):
        await scheduler.start()
        assert scheduler._task is not None
        assert not scheduler._task.done()
        await scheduler.stop()
        assert scheduler._task.done()
