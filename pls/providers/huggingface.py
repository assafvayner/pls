from __future__ import annotations

import httpx

from pls.providers import ProviderError

HF_INFERENCE_URL = "https://router.huggingface.co/v1/chat/completions"


class HuggingFaceProvider:
    def __init__(self, api_key: str, model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
        self.api_key = api_key
        self.model = model

    def generate(self, system_prompt: str, user_message: str) -> str:
        try:
            response = httpx.post(
                HF_INFERENCE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 512,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ProviderError("Invalid Hugging Face token.")
            if e.response.status_code == 429:
                raise ProviderError("Hugging Face rate limit hit. Wait a moment and try again.")
            raise ProviderError(
                f"Hugging Face error: {e.response.status_code} — {e.response.text}"
            )
        except (KeyError, IndexError):
            raise ProviderError("Unexpected response format from Hugging Face")
        except httpx.TimeoutException:
            raise ProviderError("Hugging Face request timed out.")
