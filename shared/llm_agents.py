import asyncio
import json
import os
import re
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

import anthropic
import openai
from pydantic import BaseModel
import litellm
litellm.suppress_debug_info = True
litellm.register_model(model_cost=
"https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json")

from dotenv import load_dotenv
load_dotenv()
TIMEOUT=3600
print(f"TIMEOUT: {TIMEOUT}")

def get_llm_agent_class(model: str, generation_config: dict = {}, **kwargs):
  provider, model_name = model.split("/", 1)

  provider_to_class = {
    'openai': OpenAIAgent,
    'anthropic': AnthropicAgent,
    'gemini': GeminiAgent,
    'xai': GrokAgent,
    'openrouter': OpenRouterAgent,    
  }
  assert provider in provider_to_class, f"Provider {provider} not supported"
  return provider_to_class[provider](model=model_name, **generation_config, **kwargs)


class TokenUsage(BaseModel):
  input_tokens: int = 0
  output_tokens: int = 0
  total_tokens: int = 0
  cached_tokens: int = 0
  cost: float = 0.0

class LLMResponse(BaseModel):
  content: str | None = None
  token_usage: TokenUsage | None = None
  raw: dict | None = None


class LLMAgent(ABC):
  def __init__(self, model: str, provider: str = None):
    self.model = model
    self.provider = provider
    self.all_token_usage = TokenUsage()
    self.max_token_usage = TokenUsage()
    self._usage_lock = asyncio.Lock()  # Lock for async usage updates

  def _update_usage(self, token_usage: TokenUsage):
    self.all_token_usage = sum_token_usage([self.all_token_usage, token_usage])
    self.max_token_usage = get_max_token_usage([self.max_token_usage, token_usage])

  async def _update_usage_async(self, token_usage: TokenUsage):
    """Update token usage asynchronously with lock (for concurrent async calls)."""
    async with self._usage_lock:
      self.all_token_usage = sum_token_usage([self.all_token_usage, token_usage])
      self.max_token_usage = get_max_token_usage([self.max_token_usage, token_usage])
    
  def _calculate_cost(self, response: Any) -> float:
    # Add total_tokens field if missing (e.g., Anthropic responses don't have this)
    usage = getattr(response, 'usage', None)
    if usage and not hasattr(usage, 'total_tokens'):
      usage.total_tokens = getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)
    
    try:
      cost = litellm.cost_calculator.completion_cost(completion_response=response, custom_llm_provider=self.provider)
    except Exception:
      cost = 0.0
    return cost

  @abstractmethod
  def _completions(self, messages) -> LLMResponse:
    raise NotImplementedError

  @abstractmethod
  async def _async_completions(self, messages) -> LLMResponse:
    raise NotImplementedError

  def completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    return self._completions(messages, **kwargs)

  async def async_completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    return await self._async_completions(messages, **kwargs)

class OpenAIAgent(LLMAgent):
  def __init__(self, 
               model: str,
               api_key_env: str = 'OPENAI_API_KEY',
               api_base_url: str = os.getenv('OPENAI_API_BASE_URL', 'https://api.openai.com/v1'),
               provider: str = 'openai',
               **generation_config):
    super().__init__(model=model, provider=provider)

    api_key = os.getenv(api_key_env)
    if not api_key:
      raise ValueError(f"API key not found in environment variable {api_key_env}")

    self.client = openai.OpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
    self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
    self.generation_config = generation_config

  def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
    return messages

  def _parse_response(self, response):
    """Parse response and extract content, token usage, and cost."""
    content = response.choices[0].message.content
    usage = response.usage
    cached_tokens = getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', 0)

    cost = self._calculate_cost(response)
    token_usage = TokenUsage(
      input_tokens=usage.prompt_tokens,
      output_tokens=usage.completion_tokens,
      total_tokens=usage.total_tokens,
      cached_tokens=cached_tokens,
      cost=cost,
    )

    # Convert response to dict for raw logging
    raw_response = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
    
    return content, token_usage, raw_response

  def _completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    messages = self._preprocess_messages(messages)
    response = self.client.chat.completions.create(
      model=self.model,
      messages=messages,
      **self.generation_config,
      **kwargs,
    )
    content, token_usage, raw_response = self._parse_response(response)
    self._update_usage(token_usage)
    
    return LLMResponse(content=content, token_usage=token_usage, raw=raw_response)

  async def _async_completions(self, messages: list[dict], **kwargs) -> LLMResponse:
    messages = self._preprocess_messages(messages)
    response = await self.async_client.chat.completions.create(
      model=self.model,
      messages=messages,
      **self.generation_config,
      **kwargs,
    )
    content, token_usage, raw_response = self._parse_response(response)
    await self._update_usage_async(token_usage)
    
    return LLMResponse(content=content, token_usage=token_usage, raw=raw_response)


class GrokAgent(OpenAIAgent):
  def __init__(self, model: str,
               api_key_env: str = 'XAI_API_KEY',
               api_base_url: str = 'https://api.x.ai/v1', 
               provider: str = 'xai',
               **generation_config):
    super().__init__(model=model, 
                     api_key_env=api_key_env, 
                     api_base_url=api_base_url, 
                     provider=provider,
                     **generation_config)
    # Check if this is a grok-4 model that needs conversation flattening
    self.needs_flattening = 'grok-4' in model.lower() and "grok-4-fast" not in model.lower()
  
  def _flatten_conversation(self, messages: list[dict]) -> list[dict]:
    """Flatten multi-turn conversation for Grok-4 model. This is because Grok-4 constantly give errors on multi-turn conversations.
    
    Converts user/assistant pairs into single message with USER:/ASSISTANT: markers.
    """
    if not messages:
      return messages
    
    # Separate system from conversation messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    conv_msgs = [m for m in messages if m["role"] != "system"]
    
    # No flattening needed for N message yet
    if len(conv_msgs) <= 4:
      return messages
    
    # Extract text content, handling both string and list formats
    def get_text(content):
      if isinstance(content, str):
        return content
      if isinstance(content, list):
        return " ".join(item.get("text", str(item)) for item in content if isinstance(item, dict))
      return str(content)
    
    # Build flattened conversation with USER:/ASSISTANT: markers
    flattened = "\n\n".join(f"{m['role'].upper()}: {get_text(m['content'])}" for m in conv_msgs)
    
    return system_msgs + [{"role": "user", "content": flattened}]
  
  def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
    """Preprocess messages, applying flattening for Grok-4 models."""
    if self.needs_flattening:
      return self._flatten_conversation(messages)
    return messages
 
class GeminiAgent(OpenAIAgent):
  def __init__(self, model: str, 
               api_key_env: str = 'GEMINI_API_KEY',
               api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
               provider: str = 'gemini',
               vertexai: bool = False,
               **generation_config):

    if not vertexai:
        super().__init__(model=model,
                         api_key_env=api_key_env, 
                         api_base_url=api_base_url, 
                         provider=provider,
                         **generation_config)
    else:
        # https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/chat-completions/intro_chat_completions_api.ipynb
        from google.auth import default
        from google.auth.transport.requests import Request
        
        if not os.environ.get("GOOGLE_CLOUD_PROJECT") or not os.environ.get("GOOGLE_CLOUD_LOCATION"):
            raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables must be set")
        
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        
        api_host = f"{location}-aiplatform.googleapis.com" if location != "global" else "aiplatform.googleapis.com"
        api_base_url = f"https://{api_host}/v1/projects/{project_id}/locations/{location}/endpoints/openapi"
        api_key = credentials.token
        model = f"google/{model}"
        
        self.client = openai.OpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base_url, timeout=TIMEOUT)
        self.generation_config = generation_config

class AnthropicAgent(LLMAgent):
  def __init__(self, model: str,
               use_cache: bool = False,
               vertexai: bool = False,
               provider: str = 'anthropic',
               **generation_config):
    # Determine provider before parent init
    provider = 'vertex_ai' if vertexai else provider
    super().__init__(model=model, provider=provider)

    if vertexai:
      # pip install --upgrade anthropic[vertexai]
      region = os.getenv('GOOGLE_CLOUD_LOCATION', 'global')
      project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
      assert project_id, "GOOGLE_CLOUD_PROJECT environment variable must be set for Vertex AI"
      self.client = anthropic.AnthropicVertex(region=region, project_id=project_id, timeout=TIMEOUT)
      self.async_client = anthropic.AsyncAnthropicVertex(region=region, project_id=project_id, timeout=TIMEOUT)
    else:
      assert os.getenv('ANTHROPIC_API_KEY'), "ANTHROPIC_API_KEY environment variable not set"      
      self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=TIMEOUT)
      self.async_client = anthropic.AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=TIMEOUT)
    
    # Extract cache setting from generation_config
    self.use_cache = use_cache
    self.generation_config = generation_config

  def _preprocess_messages(self, messages: list[dict]) -> list[dict]:
    system = None
    caching_messages = []
    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    # As of June 2025, Anthropic does not support auto-caching but need to define the final block as caching
    for i, message in enumerate(messages):
      if message["role"] == "system":
        system = [dict(type="text", text=message["content"])]
      else:
        new_block = {"role": message["role"]}
        
        # Handle both string content and multimodal content (list)
        if isinstance(message["content"], str):
          # Simple text message
          if self.use_cache and i == len(messages) - 1:
            content = [dict(type="text", text=message['content'], cache_control={"type": "ephemeral"})]
          else:
            content = [dict(type="text", text=message['content'])]
        elif isinstance(message["content"], list):
          # Multimodal message with text and images
          content = []
          for item in message["content"]:
            if item["type"] == "text":
              # Add cache_control to the last text block of the last message (only if caching enabled)
              if self.use_cache and i == len(messages) - 1 and item == message["content"][-1]:
                content.append(dict(type="text", text=item["text"], cache_control={"type": "ephemeral"}))
              else:
                content.append(dict(type="text", text=item["text"]))
            elif item["type"] == "image_url":
              # Convert OpenAI-style image_url to Anthropic format
              image_url = item["image_url"]["url"]
              if image_url.startswith("data:image/"):
                # Extract media type and base64 data
                media_type, base64_data = image_url.split(";base64,")
                media_type = media_type.replace("data:", "")
                content.append({
                  "type": "image",
                  "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data
                  }
                })
        else:
          # Fallback for unexpected content types
          if self.use_cache and i == len(messages) - 1:
            content = [dict(type="text", text=str(message['content']), cache_control={"type": "ephemeral"})]
          else:
            content = [dict(type="text", text=str(message['content']))]
        
        new_block["content"] = content
        caching_messages.append(new_block)
    
    return system, caching_messages

  def _parse_response(self, response):
    """Parse response and extract content, token usage, and cost."""
    content = response.content[-1].text
    usage = response.usage
    cost = self._calculate_cost(response)

    token_usage = TokenUsage(
      input_tokens=usage.input_tokens + usage.cache_creation_input_tokens,
      output_tokens=usage.output_tokens,
      cached_tokens=usage.cache_read_input_tokens,
      total_tokens=usage.input_tokens + usage.output_tokens,
      cost=cost,
    )
    
    # Convert response to dict for raw logging
    raw_response = response.model_dump() if hasattr(response, 'model_dump') else response.dict()
    
    return content, token_usage, raw_response

  def _completions(self, messages: list[dict]) -> str:
    system, messages = self._preprocess_messages(messages)
    kwargs = {
      "model": self.model,
      "messages": messages,
      **self.generation_config
    }
    if system is not None:
      kwargs["system"] = system
    
    response = self.client.messages.create(**kwargs)
    content, token_usage, raw_response = self._parse_response(response)
    self._update_usage(token_usage)
    
    return LLMResponse(content=content, token_usage=token_usage, raw=raw_response)

  async def _async_completions(self, messages: list[dict]) -> LLMResponse:
    system, messages = self._preprocess_messages(messages)
    kwargs = {
      "model": self.model,
      "messages": messages,
      **self.generation_config
    }
    if system is not None:
      kwargs["system"] = system
    
    response = await self.async_client.messages.create(**kwargs)
    content, token_usage, raw_response = self._parse_response(response)
    await self._update_usage_async(token_usage)
    
    return LLMResponse(content=content, token_usage=token_usage, raw=raw_response)

class OpenRouterAgent(OpenAIAgent):
  def __init__(self, model: str,
               api_key_env: str = 'OPENROUTER_API_KEY',
               api_base_url: str = 'https://openrouter.ai/api/v1',
               provider: str = 'openrouter',
               **generation_config):
    super().__init__(model=model,
                     api_key_env=api_key_env,
                     api_base_url=api_base_url,
                     provider=provider,
                     **generation_config)

# =================== Utils ===================
def get_agent_config(model: str, models_config_path: str = "configs/models.yaml") -> Dict[str, Any]:
    """
    Load model configuration and format it for llm_agents.
    Expected Config files to be something like this:
    ```
    gpt-5:
      model: openai/gpt-5
      generation_config:
        reasoning_effort: high
    ```
    
    Args:
        model: Model name to load
        models_config_path: Path to models configuration YAML file
        
    Returns:
        Dictionary with agent configuration {model: str, generation_config: dict}
    """
    with open(models_config_path, "r") as f:
        model_configs = yaml.safe_load(f)

    assert model in model_configs, f"Model '{model}' not found in {models_config_path}. Available models: {list(model_configs.keys())}"
    print("==== Model config: ", model_configs[model])
    return model_configs[model]

def sum_token_usage(token_usages: list[TokenUsage]):
  input_tokens = sum(t.input_tokens for t in token_usages)
  output_tokens = sum(t.output_tokens for t in token_usages)
  total_tokens = sum(t.total_tokens for t in token_usages)
  cached_tokens = sum(t.cached_tokens for t in token_usages)
  cost = sum(t.cost for t in token_usages)
  return TokenUsage(input_tokens=input_tokens, 
                    output_tokens=output_tokens, 
                    total_tokens=total_tokens, 
                    cached_tokens=cached_tokens,
                    cost=cost)

def get_max_token_usage(token_usages: list[TokenUsage]):
  return TokenUsage(input_tokens=max(t.input_tokens for t in token_usages), 
                    output_tokens=max(t.output_tokens for t in token_usages), 
                    total_tokens=max(t.total_tokens for t in token_usages), 
                    cached_tokens=max(t.cached_tokens for t in token_usages),
                    cost=max(t.cost for t in token_usages))