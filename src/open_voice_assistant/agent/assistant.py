"""AI agent using OpenAI Agents SDK with streaming."""

import logging
from collections.abc import AsyncIterator

from agents import Agent, RunConfig, Runner, SQLiteSession
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.stream_events import RawResponsesStreamEvent
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from open_voice_assistant.config import Settings

logger = logging.getLogger(__name__)

_RUN_CONFIG = RunConfig(tracing_disabled=True)


class Assistant:
    """Conversational agent backed by OpenAI Agents SDK with token streaming."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._agent: Agent | None = None
        self._session: SQLiteSession | None = None

    def _build_model(self) -> OpenAIChatCompletionsModel:
        client = AsyncOpenAI(
            base_url=self._settings.openai_base_url or "https://api.openai.com/v1",
            api_key=self._settings.openai_api_key or "not-set",
        )
        return OpenAIChatCompletionsModel(
            model=self._settings.agent_model,
            openai_client=client,
        )

    def load(self) -> None:
        logger.info("Initializing agent with model: %s", self._settings.agent_model)
        if self._settings.openai_base_url:
            logger.info("  Base URL: %s", self._settings.openai_base_url)

        self._agent = Agent(
            name="voice-assistant",
            instructions=self._settings.agent_instructions,
            model=self._build_model(),
        )
        logger.info("Agent initialized")

    async def reset_history(self) -> None:
        """Clear conversation history. Call at the start of a new wake word session."""
        if self._session:
            await self._session.clear_session()
        self._session = SQLiteSession("voice")
        logger.debug("Conversation history reset")

    async def run_streamed(self, text: str) -> AsyncIterator[str]:
        """Run the agent with session history and yield content tokens."""
        if self._agent is None:
            raise RuntimeError("Call load() first")
        if self._session is None:
            self._session = SQLiteSession("voice")

        logger.debug("Agent processing: %r", text[:80])

        try:
            result = Runner.run_streamed(
                self._agent,
                input=text,
                run_config=_RUN_CONFIG,
                session=self._session,
            )

            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                if not isinstance(event.data, ResponseTextDeltaEvent):
                    continue
                yield event.data.delta
        except Exception:
            logger.exception("Agent call failed")
            yield "Sorry, I could not process that."

        logger.debug("Agent response complete")
