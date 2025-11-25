"""
Base agent class for Mindcore AI agents.
"""
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from openai import OpenAI, APIError, RateLimitError, APIConnectionError, APITimeoutError

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AgentInitializationError(Exception):
    """Raised when agent initialization fails."""
    pass


class APICallError(Exception):
    """Raised when API call fails after all retries."""
    pass


class BaseAgent(ABC):
    """
    Abstract base class for AI agents.

    Provides common functionality for OpenAI API interactions.
    """

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0  # Base delay in seconds
    RETRY_DELAY_MAX = 30.0  # Maximum delay in seconds
    REQUEST_TIMEOUT = 60  # Timeout in seconds

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.3, max_tokens: int = 1000):
        """
        Initialize base agent.

        Args:
            api_key: OpenAI API key.
            model: Model name (default: gpt-4o-mini).
            temperature: Temperature for generation.
            max_tokens: Maximum tokens in response.

        Raises:
            AgentInitializationError: If API key is missing or invalid.
        """
        if not api_key or not isinstance(api_key, str) or len(api_key.strip()) == 0:
            raise AgentInitializationError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide api_key in config.yaml"
            )

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, timeout=self.REQUEST_TIMEOUT)
        logger.info(f"Initialized {self.__class__.__name__} with model {model}")

    def _call_openai(
        self,
        messages: list,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Call OpenAI API with automatic retry logic.

        Args:
            messages: List of message dictionaries.
            response_format: Optional response format specification.
            temperature: Override temperature.
            max_tokens: Override max_tokens.

        Returns:
            Response content as string.

        Raises:
            APICallError: If all retries fail.
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                logger.debug(f"OpenAI API call successful: {len(content)} chars")
                return content

            except RateLimitError as e:
                last_exception = e
                delay = min(self.RETRY_DELAY_BASE * (2 ** attempt), self.RETRY_DELAY_MAX)
                logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(delay)

            except (APIConnectionError, APITimeoutError) as e:
                last_exception = e
                delay = min(self.RETRY_DELAY_BASE * (2 ** attempt), self.RETRY_DELAY_MAX)
                logger.warning(f"Connection error, retrying in {delay}s (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                time.sleep(delay)

            except APIError as e:
                # Retry on 5xx server errors
                if hasattr(e, 'status_code') and e.status_code >= 500:
                    last_exception = e
                    delay = min(self.RETRY_DELAY_BASE * (2 ** attempt), self.RETRY_DELAY_MAX)
                    logger.warning(f"Server error {e.status_code}, retrying in {delay}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    # Don't retry client errors (4xx)
                    logger.error(f"OpenAI API client error: {e}")
                    raise APICallError(f"OpenAI API call failed: {e}") from e

            except Exception as e:
                # Unknown error, don't retry
                logger.error(f"OpenAI API call failed with unexpected error: {e}")
                raise APICallError(f"OpenAI API call failed: {e}") from e

        # All retries exhausted
        logger.error(f"OpenAI API call failed after {self.MAX_RETRIES} retries: {last_exception}")
        raise APICallError(f"OpenAI API call failed after {self.MAX_RETRIES} retries") from last_exception

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from OpenAI.

        Args:
            response: Response string.

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If response cannot be parsed as JSON.
        """
        if not response or not isinstance(response, str):
            raise ValueError("Empty or invalid response from OpenAI")

        original_response = response

        try:
            # Try to extract JSON from code blocks if present
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end > start:
                    response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end > start:
                    response = response[start:end].strip()

            # Try parsing the extracted/original response
            return json.loads(response)

        except json.JSONDecodeError as e:
            # Try to find JSON object in the response
            try:
                # Look for JSON object pattern
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end > start:
                    json_str = response[start:end + 1]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {original_response[:500]}...")
            raise ValueError(
                f"Failed to parse LLM response as JSON. Response started with: "
                f"{original_response[:100]}..."
            ) from e

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Process method to be implemented by subclasses.

        Returns:
            Processing result.
        """
        pass
