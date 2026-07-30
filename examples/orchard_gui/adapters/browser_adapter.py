"""
Browser environment adapter.

This module provides the adapter for integrating browser with slime's
training pipeline using the standard gym interface.
"""

import time
import base64
import json
import re
from dataclasses import dataclass
from typing import Any
from examples.orchard_gui.base.adapter import BaseGymEnvAdapter
from examples.orchard_gui.base.types import EnvConfig, ParsedToolResult, ToolCall

import logging

logger = logging.getLogger(__name__)


@dataclass
class BrowserEnvConfig(EnvConfig):
    """Configuration specific to browser environment."""
    # Environment mode: "local", "remote", "sandbox", or "browser-use"
    mode: str = "local"
    # Sandbox-specific config (used when mode: sandbox)
    sandbox: dict | None = None
    # Browser-Use-specific config (used when mode: browser-use)
    browser_use: dict | None = None
    # Remote-specific config (used when mode: remote)
    remote: dict | None = None

    # Process-wide cap on concurrent env creation (see env/factory.py).
    # <= 0 disables it.
    max_env_concurrency: int = 0

    # Web browser environment configuration
    width: int = 1280
    height: int = 720
    dpr: int = 1

    max_retries: int = 3
    wait_timeout: int = 5

    resize_output_coords: bool = True
    resize_scale: int = 1000
    image_patch_size: int = 32 # not used if resize_scale > 0

    path_to_tool_list: str | None = None
    path_to_policy: str | None = None

    # Additional configuration for browser adapter
    use_screenshot: bool = True
    use_a11ytree: bool = False


class BrowserAdapter(BaseGymEnvAdapter):
    """
    Adapter for browser gymnasium environment.
    
    Browser provides multi-domain task-oriented dialogue benchmarks
    with a gymnasium-compatible interface for agent training.
    """
    
    def __init__(self, config: BrowserEnvConfig, rollout_args: Any):
        """
        Initialize the browser adapter.
        
        Args:
            config: Browser-specific configuration
            rollout_args: Rollout arguments from slime
        """
        super().__init__(config, rollout_args)

    async def create_env(self, task_id: str) -> Any:
        """
        Create and return the gym environment.
        
        Args:
            task_id: Task identifier for the environment
            
        Returns:
            Gymnasium-compatible environment instance
        """
        pass

    def parse_env_info(self, info: dict[str, Any]) -> tuple[list[dict], str]:
        """
        Extract tools and policy from browser env info.
        
        Args:
            info: Info dict from env.reset()
            
        Returns:
            Tuple of (tools_info, policy)
        """
        tool_list = info.get("tool_list", [])
        policy = info.get("policy", "")
        return tool_list, policy

    def process_observation(
        self, 
        observation: str | None, 
        parsed_result: ParsedToolResult,
        messages: list[dict],
    ) -> dict[str, Any] | None:
        """
        Convert environment observation to a message.
        
        Args:
            observation: Raw observation from environment step
            parsed_result: Parsed tool call result from LLM response
            messages: Current conversation messages
            
        Returns:
            Message dict to append to conversation, or None if no message
        """
        pass

    def _build_tab_observation_text(self, all_tabs: list[dict[str, Any]]) -> str:
        """Build tab information text from the tab list in observation."""
        lines = []
        active_tab_index = next((tab["index"] for tab in all_tabs if tab.get("active")), None)

        if active_tab_index is not None:
            lines.append(f"Current tab: {active_tab_index}")

        lines.append("Available tabs:")
        for tab in all_tabs:
            idx = tab["index"]
            url = tab.get("url", "")
            title = tab.get("title", "")
            marker = " (active)" if tab.get("active") else ""
            tab_desc = f"- Tab {idx}{marker}: {url}"
            if title:
                tab_desc += f" - {title}"
            lines.append(tab_desc)

        return "\n".join(lines)

    def render_observation(self, observation: dict[str, Any], step_id: int = None) -> tuple[str, str | None]:
        """Render an environment observation into model-facing text plus an
        optional screenshot data-URL.

        Returns:
            (ob_text, ob_screen_base64): ``ob_text`` is the ``<observation>...``
            block (tab info, optional a11ytree, screenshot placeholder);
            ``ob_screen_base64`` is the ``data:image/png;base64,...`` screenshot
            or ``None`` when no screenshot is attached.
        """
        ob_screen_base64 = None
        if not observation:
            return "<observation>\n(no observation available)\n</observation>", None

        screen_size = observation.get("screen_size") or (0, 0)
        ob_text = f"""<observation>\nscreen size: {screen_size[0]} x {screen_size[1]}\n{self._build_tab_observation_text(observation.get("all_tab_url", []))}"""

        screenshot, a11ytree = None, None

        if self.config.use_a11ytree and observation:
            a11ytree = json.dumps(observation.get("a11ytree", None))
            # append the a11ytree to the text message
            ob_text += f"\na11ytree: {a11ytree}"

        if self.config.use_screenshot and observation:
            screenshot = observation.get("screenshot", None)
            if not isinstance(screenshot, bytes):
                raise TypeError(f"Screenshot must be bytes, got {type(screenshot)}")
            screenshot = base64.b64encode(screenshot).decode('utf-8')
            # save the screenshot to a file
            # with open(f"examples/orchard_gui/screenshots/screenshot_{str(time.time())}_{step_id}.png", "wb") as f:
            #     f.write(base64.b64decode(screenshot))
            # append the screenshot placeholder to the text message
            ob_text += f"\nscreenshot:\n<|vision_start|><|image_pad|><|vision_end|>"
            # append the screenshot to the messages
            ob_screen_base64 = f"data:image/png;base64,{screenshot}"

        # append the observation end to the text message
        ob_text += "\n</observation>"
        return ob_text, ob_screen_base64

    def clean_tool_response(self, response: str) -> str:
        """Clean tool response text so it is concise and well-formatted for the
        next turn's input (truncate overly long responses, shorten media tags)."""
        cleaned = response.strip()
        if len(cleaned) > 1000:
            cleaned = cleaned[:1000] + "... (truncated)"

        cleaned = (
            cleaned
            .replace("<video>", "<vid>").replace("</video>", "</vid>")
            .replace("<image>", "<img>").replace("</image>", "</img>")
            .replace("<audio>", "<aud>").replace("</audio>", "</aud>")
        )
        return cleaned

    def get_initial_messages(self, user_request: str, policy: str, initial_observation: str | None) -> list[dict[str, Any]]:
        """
        Build initial conversation messages for browser environment.
        
        Args:
            user_request: User's initial request
            policy: Domain policy from environment
            initial_observation: Initial user query
            
        Returns:
            List of system and user messages
        """
        ob_text, ob_screen_base64 = self.render_observation(initial_observation)
        messages = [
            {
                "role": "system",
                "content": policy
            },
            {
                "role": "user",
                "content": f"<user_request>\n{user_request}\n</user_request>\n{ob_text}",
            },
        ]
        return messages, ob_screen_base64 if ob_screen_base64 else None

    def actions_from_parsed(self, parsed_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """
        Convert parsed tool result to list of actions.
        For single action parse, please refer to self.action_from_parsed().
        Args:
            parsed_calls: List of tool calls from LLM response
            
        Returns:
            List of actions
        """
        actions = []
        for tool_call in parsed_calls:
            params = (
                json.loads(tool_call.parameters) 
                if isinstance(tool_call.parameters, str) 
                else tool_call.parameters
            )
            actions.append({
                "name": tool_call.name,
                "args": params
            })
        return actions
    
    async def _run_inference_step(
        self,
        url: str,
        tokens: list[int],
        sampling_params: dict,
        image_data,
    ):
        """
        Run a single inference step.
        """
        pass