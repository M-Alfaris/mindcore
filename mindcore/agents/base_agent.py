"""
Base agent class for Mindcore AI agents.
"""
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from openai import OpenAI

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.

    Provides common functionality for OpenAI API interactions.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3, max_tokens: int = 1000):
        """
        Initialize base agent.

        Args:
            api_key: OpenAI API key.
            model: Model name (default: gpt-4o-mini).
            temperature: Temperature for generation.
            max_tokens: Maximum tokens in response.
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized {self.__class__.__name__} with model {model}")

    def _call_openai(
        self,
        messages: list,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call OpenAI API.

        Args:
            messages: List of message dictionaries.
            response_format: Optional response format specification.
            temperature: Override temperature.
            max_tokens: Override max_tokens.

        Returns:
            Response content as string.
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }

            if response_format:
                kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            logger.debug(f"OpenAI API call successful: {len(content)} chars")
            return content

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from OpenAI.

        Args:
            response: Response string.

        Returns:
            Parsed JSON dictionary.
        """
        try:
            # Try to extract JSON from code blocks if present
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end].strip()

            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            raise

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Process method to be implemented by subclasses.

        Returns:
            Processing result.
        """
        pass
