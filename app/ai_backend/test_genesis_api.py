import asyncio
import aiohttp
import json
import re
from typing import Any, Callable, Dict, Optional

class GenesisAPIError(Exception):
    pass

class GenesisAPIAuthenticationError(GenesisAPIError):
    pass

class GenesisAPIRateLimitError(GenesisAPIError):
    pass

class GenesisAPIServerError(GenesisAPIError):
    pass

class GenesisAPITimeoutError(GenesisAPIError):
    pass

def validate_api_key(api_key: Optional[str]) -> bool:
    if not api_key or not isinstance(api_key, str):
        return False
    if not api_key.startswith("sk-"):
        return False
    key_body = api_key[3:]
    if len(key_body) < 10:
        return False
    if not re.fullmatch(r"[A-Za-z0-9]+", key_body):
        return False
    return True

def format_genesis_prompt(prompt: str, model: str = "genesis-v1", max_tokens: int = 256, temperature: float = 1.0) -> Dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise ValueError("Model must be a non-empty string")
    if not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    if not isinstance(temperature, (int, float)) or not (0 <= temperature <= 1):
        raise ValueError("temperature must be between 0 and 1")
    return {
        "prompt": prompt,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

class GenesisRequest:
    def __init__(self, prompt: str, model: str, max_tokens: int = 256, temperature: float = 1.0):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or not (0 <= temperature <= 1):
            raise ValueError("temperature must be between 0 and 1")
        self.prompt = prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenesisRequest":
        return cls(
            prompt=data.get("prompt"),
            model=data.get("model"),
            max_tokens=data.get("max_tokens", 256),
            temperature=data.get("temperature", 1.0)
        )

class GenesisResponse:
    def __init__(self, id: str, text: str, model: str, created: int, usage: Dict[str, Any]):
        self.id = id
        self.text = text
        self.model = model
        self.created = created
        self.usage = usage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "model": self.model,
            "created": self.created,
            "usage": self.usage
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenesisResponse":
        return cls(
            id=data["id"],
            text=data["text"],
            model=data["model"],
            created=data["created"],
            usage=data["usage"]
        )

def parse_genesis_response(response: Dict[str, Any]) -> GenesisResponse:
    if not isinstance(response, dict):
        raise GenesisAPIError("Response is not a valid JSON object")
    for field in ("id", "text", "model", "created"):
        if field not in response:
            raise GenesisAPIError(f"Missing field {field} in response")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise GenesisAPIError("Invalid usage format in response")
    return GenesisResponse.from_dict(response)

def handle_genesis_error(status_code: int, error_body: Dict[str, Any]) -> None:
    message = ""
    if isinstance(error_body, dict):
        message = error_body.get("error", "")
    if status_code == 401:
        raise GenesisAPIAuthenticationError(message)
    elif status_code == 429:
        raise GenesisAPIRateLimitError(message)
    elif 500 <= status_code < 600:
        raise GenesisAPIServerError(message)
    else:
        raise GenesisAPIError(message)

async def retry_with_exponential_backoff(func: Callable[..., Any], max_retries: int = 3) -> Any:
    attempt = 0
    while True:
        try:
            return await func()
        except GenesisAPIAuthenticationError:
            raise
        except GenesisAPIServerError:
            attempt += 1
            if attempt >= max_retries:
                raise
            await asyncio.sleep(2 ** attempt)
        except GenesisAPIError:
            raise

class GenesisAPI:
    def __init__(self, api_key: str, base_url: str, timeout: int = 30, max_retries: int = 3):
        if not validate_api_key(api_key):
            raise GenesisAPIAuthenticationError("Invalid API key")
        if not base_url or not isinstance(base_url, str):
            raise ValueError("Invalid base URL")
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def generate_text(self, prompt: str, model: str = "genesis-v1", max_tokens: int = 256, temperature: float = 1.0) -> GenesisResponse:
        if prompt is None or not isinstance(prompt, str) or not prompt:
            raise ValueError("Prompt must be a non-empty string")
        payload = format_genesis_prompt(prompt, model, max_tokens, temperature)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        async def do_request():
            try:
                timeout_conf = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout_conf) as session:
                    async with session.post(self.base_url, headers=headers, json=payload) as resp:
                        status = resp.status
                        try:
                            body = await resp.json()
                        except json.JSONDecodeError as e:
                            raise GenesisAPIError(str(e))
                        if status == 200:
                            return parse_genesis_response(body)
                        else:
                            handle_genesis_error(status, body)
            except asyncio.TimeoutError:
                raise GenesisAPITimeoutError("Request timed out")
            except aiohttp.ClientError as e:
                raise GenesisAPIError(str(e))
        return await retry_with_exponential_backoff(do_request, max_retries=self.max_retries)