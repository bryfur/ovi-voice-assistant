"""AI agent using OpenAI Agents SDK with streaming and MCP tool support."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from agents import Agent, RunConfig, Runner, SQLiteSession
from agents.mcp import MCPServerStdio
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.stream_events import RawResponsesStreamEvent
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from ovi_voice_assistant.agent.assistant_context import AssistantContext
from ovi_voice_assistant.agent.tools import BUILTIN_TOOLS
from ovi_voice_assistant.config import Settings

logger = logging.getLogger(__name__)

_RUN_CONFIG = RunConfig(tracing_disabled=True)


class Assistant:
    """Conversational agent backed by OpenAI Agents SDK with token streaming."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._agent: Agent | None = None
        self._session: SQLiteSession | None = None
        self._mcp_servers: list[MCPServerStdio] = []

    def _build_model(self) -> OpenAIChatCompletionsModel:
        client = AsyncOpenAI(
            base_url=self._settings.openai_base_url or "https://api.openai.com/v1",
            api_key=self._settings.openai_api_key or "not-set",
        )
        return OpenAIChatCompletionsModel(
            model=self._settings.agent_model,
            openai_client=client,
        )

    def _parse_mcp_servers(self) -> list[MCPServerStdio]:
        raw = self._settings.mcp_servers.strip()
        if not raw:
            return []

        if raw.startswith("@"):
            with open(raw[1:]) as f:
                entries = json.load(f)
        else:
            entries = json.loads(raw)
        servers = []
        for entry in entries:
            name = entry.get(
                "name", entry["args"][-1] if entry.get("args") else entry["command"]
            )
            servers.append(
                MCPServerStdio(
                    name=name,
                    params={"command": entry["command"], "args": entry.get("args", [])},
                    cache_tools_list=True,
                )
            )
        return servers

    def _parse_agents(
        self, model: OpenAIChatCompletionsModel
    ) -> list[tuple[Agent, list[MCPServerStdio], str]]:
        """Parse sub-agent definitions from config."""
        raw = self._settings.agents.strip()
        if not raw:
            return []

        if raw.startswith("@"):
            with open(raw[1:]) as f:
                entries = json.load(f)
        else:
            entries = json.loads(raw)

        result = []
        for entry in entries:
            servers = []
            for srv in entry.get("mcp_servers", []):
                name = srv.get(
                    "name", srv["args"][-1] if srv.get("args") else srv["command"]
                )
                servers.append(
                    MCPServerStdio(
                        name=name,
                        params={"command": srv["command"], "args": srv.get("args", [])},
                        cache_tools_list=True,
                    )
                )
            agent = Agent(
                name=entry["name"],
                instructions=entry.get("instructions", ""),
                model=model,
                mcp_servers=servers,
            )
            result.append((agent, servers, entry.get("description", "")))
        return result

    def load(self) -> None:
        logger.info("Initializing agent with model: %s", self._settings.agent_model)
        if self._settings.openai_base_url:
            logger.info("  Base URL: %s", self._settings.openai_base_url)

        self._mcp_servers = self._parse_mcp_servers()
        if self._mcp_servers:
            logger.info("  MCP servers: %s", [s.name for s in self._mcp_servers])

        model = self._build_model()

        sub_agent_tools = []
        for agent, servers, description in self._parse_agents(model):
            self._mcp_servers.extend(servers)
            sub_agent_tools.append(
                agent.as_tool(
                    tool_name=agent.name,
                    tool_description=description,
                )
            )
            logger.info("  Sub-agent: %s", agent.name)

        self._agent = Agent[AssistantContext](
            name="voice-assistant",
            instructions=self._settings.agent_instructions,
            model=model,
            tools=[*BUILTIN_TOOLS, *sub_agent_tools],
            mcp_servers=self._mcp_servers,
        )
        logger.info("Agent initialized")

    async def start(self) -> None:
        """Start MCP server connections."""
        for server in self._mcp_servers:
            await server.__aenter__()
            logger.info("MCP server started: %s", server.name)

    async def stop(self) -> None:
        """Stop MCP server connections."""
        for server in self._mcp_servers:
            try:
                await asyncio.wait_for(
                    server.__aexit__(None, None, None),
                    timeout=3.0,
                )
            except (TimeoutError, asyncio.CancelledError):
                logger.debug("MCP server %s did not stop cleanly", server.name)
            except Exception:
                logger.exception("Error stopping MCP server: %s", server.name)

    async def reset_history(self) -> None:
        """Clear conversation history. Call at the start of a new wake word session."""
        if self._session:
            await self._session.clear_session()
        self._session = SQLiteSession("voice")
        logger.debug("Conversation history reset")

    async def run_text(self, text: str, context: AssistantContext | None = None) -> str:
        """Run the agent and return the full response as a string."""
        tokens: list[str] = []
        async for token in self.run_streamed(text, context=context):
            tokens.append(token)
        return "".join(tokens)

    async def run_streamed(
        self, text: str, context: AssistantContext | None = None
    ) -> AsyncIterator[str]:
        """Run the agent with session history and yield content tokens."""
        if self._agent is None:
            raise RuntimeError("Call load() first")
        if self._session is None:
            self._session = SQLiteSession("voice")

        logger.debug("Agent processing: %r", text[:80])

        # Auto-recall: inject relevant memories into the prompt
        input_text = text
        if context and context.memory:
            input_text = await self._inject_memories(text, context)

        try:
            result = Runner.run_streamed(
                self._agent,
                input=input_text,
                run_config=_RUN_CONFIG,
                session=self._session,
                context=context,
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

    async def _inject_memories(self, text: str, context: AssistantContext) -> str:
        """Recall relevant memories and prepend them to the user's input."""
        try:
            result = await context.memory.recall(text, max_tokens=256)
            if not result.results:
                return text
            facts = "\n".join(f"- {f.text}" for f in result.results)
            print(facts)

            logger.debug("Injected %d memories", len(result.results))
            return f"[Relevant memories]\n{facts}\n\n[User]\n{text}"
        except Exception:
            logger.exception("Memory recall failed, proceeding without context")
            return text
