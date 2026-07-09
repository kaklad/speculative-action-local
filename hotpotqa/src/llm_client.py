import os

import openai
from google import genai
from google.genai import types

from . import constants


class LLMClient:
    """Unified LLM client supporting local, Gemini, OpenAI, and OpenRouter APIs."""

    def __init__(self, model_name, temperature, max_tokens, top_p, role="main"):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.role = role
        self.gemini_client = None
        self.openai_client = None
        self.openrouter_client = None
        self.local_client = None

    def call(self, prompt, stop=None):
        if self._is_local_model():
            return self._local_call(prompt, stop)
        if self.model_name.startswith("gemini"):
            return self._gemini_call(prompt, stop)
        elif self.model_name.startswith("gpt"):
            return self._openai_call(prompt, stop)
        else:
            return self._openrouter_call(prompt, stop)

    def _is_local_model(self):
        return self.model_name.startswith(("/", "./", "../")) or "models/" in self.model_name

    def _local_base_url(self):
        if self.role == "guess":
            return os.getenv("LOCAL_GUESS_BASE_URL", constants.local_guess_base_url)
        return os.getenv("LOCAL_MAIN_BASE_URL", constants.local_main_base_url)

    def _get_gemini_client(self):
        if self.gemini_client is None:
            self.gemini_client = genai.Client(api_key=constants.gemini_api_key)
        return self.gemini_client

    def _get_openai_client(self):
        if self.openai_client is None:
            self.openai_client = openai.OpenAI(api_key=constants.openai_api_key)
        return self.openai_client

    def _get_openrouter_client(self):
        if self.openrouter_client is None:
            self.openrouter_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=constants.openrouter_api_key,
            )
        return self.openrouter_client

    def _get_local_client(self):
        if self.local_client is None:
            self.local_client = openai.OpenAI(
                base_url=self._local_base_url(),
                api_key=os.getenv("LOCAL_API_KEY", constants.local_api_key),
            )
        return self.local_client

    def _gemini_call(self, prompt, stop):
        if stop is not None:
            config = types.GenerateContentConfig(stop_sequences=stop)
        else:
            config = types.GenerateContentConfig()
        response = self._get_gemini_client().models.generate_content(
            model=self.model_name, contents=prompt, config=config
        )
        return str(response.text)

    def _openai_call(self, prompt, stop):
        response = self._get_openai_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

    def _openrouter_call(self, prompt, stop):
        response = self._get_openrouter_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content

    def _local_call(self, prompt, stop):
        response = self._get_local_client().chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=stop,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response.choices[0].message.content
