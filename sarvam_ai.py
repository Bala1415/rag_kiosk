"""
Sarvam AI Client - Wrapper for Sarvam AI API calls.
"""
import os
import logging
from typing import Dict, List, Any, Optional
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

# Sarvam AI API base URL
SARVAM_API_BASE = os.getenv("SARVAM_API_BASE", "https://api.sarvam.ai/v1")


class SarvamClient:
    """Client for interacting with Sarvam AI API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is required. Set it in .env or pass to constructor.")
        
        self.session = requests.Session()
        self.session.headers.update({
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        })
    
    def chat(
        self,
        model: str = "sarvam-105b",
        messages: List[Dict[str, str]] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call Sarvam AI chat/completions API.
        
        Args:
            model: Model name (e.g., "sarvam-2-mini", "sarvam-2")
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            **kwargs: Additional parameters
            
        Returns:
            API response dict
        """
        url = f"{SARVAM_API_BASE}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages or [],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Sarvam API HTTP error: {e}")
            logger.error(f"Response: {e.response.text if e.response else 'No response'}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Sarvam API request error: {e}")
            raise
    
    def chat_stream(
        self,
        model: str = "sarvam-105b",
        messages: List[Dict[str, str]] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ):
        """Stream chat responses (if supported)."""
        url = f"{SARVAM_API_BASE}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages or [],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        try:
            response = self.session.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data != "[DONE]":
                            import json
                            yield json.loads(data)
                            
        except requests.exceptions.RequestException as e:
            logger.error(f"Sarvam API stream error: {e}")
            raise


# Backwards compatibility function
def get_sarvam_client() -> SarvamClient:
    """Get a configured SarvamClient instance."""
    return SarvamClient()