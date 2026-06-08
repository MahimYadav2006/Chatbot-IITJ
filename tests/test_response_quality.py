"""Tests for response sanitization — ensures no HTML noise in chatbot responses."""

import sys
import os
from unittest.mock import Mock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag.llm import BedrockLLM, GeminiLLM, OllamaLLM, GroqLLM, create_llm_from_env, get_system_prompt, sanitize_response


class TestSanitizeResponse:
    def test_strips_target_blank(self):
        text = 'Visit [IIT Jammu](https://iitjammu.ac.in){target="_blank"}'
        result = sanitize_response(text)
        assert 'target' not in result
        assert '_blank' not in result

    def test_strips_rel_noopener(self):
        text = 'Click here {rel="noopener"} for more info'
        result = sanitize_response(text)
        assert 'noopener' not in result

    def test_strips_combined_attributes(self):
        text = 'Link {: target="_blank" rel="noopener"}'
        result = sanitize_response(text)
        assert 'target' not in result
        assert 'noopener' not in result

    def test_converts_html_anchor_to_markdown(self):
        text = 'Visit <a href="https://iitjammu.ac.in" target="_blank" rel="noopener">IIT Jammu</a>'
        result = sanitize_response(text)
        assert '<a' not in result
        assert '</a>' not in result
        assert 'IIT Jammu' in result
        assert 'iitjammu.ac.in' in result

    def test_strips_div_span_tags(self):
        text = '<div class="info">Hello</div> <span>World</span>'
        result = sanitize_response(text)
        assert '<div' not in result
        assert '</div>' not in result
        assert '<span' not in result
        assert 'Hello' in result
        assert 'World' in result

    def test_preserves_markdown_links(self):
        text = 'Visit [IIT Jammu](https://iitjammu.ac.in) for more'
        result = sanitize_response(text)
        assert '[IIT Jammu](https://iitjammu.ac.in)' in result

    def test_cleans_attributes_inside_markdown_link_url(self):
        text = 'Visit [IIT Jammu EE](https://iitjammu.ac.in/ee" target="_blank" rel="noopener")'
        result = sanitize_response(text)
        assert result == 'Visit [IIT Jammu EE](https://iitjammu.ac.in/ee)'

    def test_cleans_malformed_anchor_fragment(self):
        text = 'Visit https://iitjammu.ac.in/ee" target="_blank" rel="noopener">IIT Jammu EE website'
        result = sanitize_response(text)
        assert 'target' not in result
        assert 'noopener' not in result
        assert 'https://iitjammu.ac.in/ee' in result
        assert 'IIT Jammu EE website' in result

    def test_preserves_plain_text(self):
        text = 'The department has 24 faculty members working in various research areas.'
        result = sanitize_response(text)
        assert result == text

    def test_preserves_email_addresses(self):
        text = 'Contact: ajay.singh@iitjammu.ac.in'
        result = sanitize_response(text)
        assert 'ajay.singh@iitjammu.ac.in' in result

    def test_strips_standalone_target_blank(self):
        text = 'target="_blank" some content rel="noopener"'
        result = sanitize_response(text)
        assert 'target' not in result.lower()
        assert 'noopener' not in result.lower()
        assert 'some content' in result


class TestOllamaLLM:
    def test_successful_response(self, monkeypatch):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "message": {
                "content": "Hello! I am Ollama."
            }
        }
        
        monkeypatch.setattr("graphrag.llm.requests.post", lambda *args, **kwargs: response)
        
        llm = OllamaLLM()
        result = llm.generate("hello")
        assert result == "Hello! I am Ollama."

    def test_timeout_fallback_message(self, monkeypatch):
        import requests
        def mock_post(*args, **kwargs):
            raise requests.exceptions.Timeout("Connection timed out")
            
        monkeypatch.setattr("graphrag.llm.requests.post", mock_post)
        monkeypatch.setattr("graphrag.llm.time.sleep", lambda *_args, **_kwargs: None)
        
        llm = OllamaLLM()
        result = llm.generate("hello")
        assert "timed out" in result.lower()

    def test_general_exception_fallback_message(self, monkeypatch):
        def mock_post(*args, **kwargs):
            raise RuntimeError("Generic connection error")
            
        monkeypatch.setattr("graphrag.llm.requests.post", mock_post)
        monkeypatch.setattr("graphrag.llm.time.sleep", lambda *_args, **_kwargs: None)
        
        llm = OllamaLLM()
        result = llm.generate("hello")
        assert "error generating a response" in result.lower()


class TestBedrockLLM:
    def test_successful_response(self, monkeypatch):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Hello! I am Bedrock."}
                    ]
                }
            }
        }

        monkeypatch.setattr("graphrag.llm.requests.post", lambda *args, **kwargs: response)

        llm = BedrockLLM(api_key="test-key", model="test-model", region="us-east-1")
        result = llm.generate("hello")
        assert result == "Hello! I am Bedrock."

    def test_retries_with_inference_profile_when_base_model_is_rejected(self, monkeypatch):
        class FakeResponse:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload
                self.text = str(payload)
                self.ok = 200 <= status_code < 300

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    import requests
                    raise requests.exceptions.HTTPError(
                        f"{self.status_code} Client Error",
                        response=self,
                    )

        calls = []

        def mock_post(url, *args, **kwargs):
            calls.append(url)
            if "model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse" in url:
                return FakeResponse(
                    400,
                    {
                        "message": (
                            "Invocation of model ID anthropic.claude-haiku-4-5-20251001-v1:0 "
                            "with on-demand throughput isn't supported. Retry your request "
                            "with the ID or ARN of an inference profile that contains this model."
                        )
                    },
                )
            if "model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse" in url:
                return FakeResponse(
                    200,
                    {
                        "output": {
                            "message": {
                                "content": [
                                    {"text": "Recovered response"}
                                ]
                            }
                        }
                    },
                )
            raise AssertionError(f"Unexpected URL {url}")

        monkeypatch.setattr("graphrag.llm.requests.post", mock_post)

        llm = BedrockLLM(
            api_key="test-key",
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            region="us-east-1",
        )
        result = llm.generate("hello")

        assert result == "Recovered response"
        assert llm.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert len(calls) == 2

    def test_missing_api_key_fallback_message(self, monkeypatch):
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "")
        llm = BedrockLLM()
        result = llm.generate("hello")
        assert "error generating a response" in result.lower()

    def test_falls_back_to_configured_backup_model_when_account_cannot_use_primary(self, monkeypatch):
        class FakeResponse:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload
                self.text = str(payload)
                self.ok = 200 <= status_code < 300

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    import requests
                    raise requests.exceptions.HTTPError(
                        f"{self.status_code} Client Error",
                        response=self,
                    )

        def mock_post(url, *args, **kwargs):
            if "model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse" in url:
                return FakeResponse(
                    404,
                    {
                        "message": (
                            "Model use case details have not been submitted for this account."
                        )
                    },
                )
            if "model/amazon.nova-micro-v1%3A0/converse" in url:
                return FakeResponse(
                    200,
                    {
                        "output": {
                            "message": {
                                "content": [
                                    {"text": "Fallback response"}
                                ]
                            }
                        }
                    },
                )
            raise AssertionError(f"Unexpected URL {url}")

        monkeypatch.setattr("graphrag.llm.requests.post", mock_post)

        llm = BedrockLLM(
            api_key="test-key",
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            region="us-east-1",
        )
        llm.fallback_model = "amazon.nova-micro-v1:0"

        result = llm.generate("hello")

        assert result == "Fallback response"
        assert llm.model == "amazon.nova-micro-v1:0"


class TestLLMFactory:
    def test_create_llm_from_env_uses_ollama(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")

        llm = create_llm_from_env()

        assert isinstance(llm, OllamaLLM)
        assert llm.model == "llama3.2"

    def test_create_llm_from_env_uses_gemini(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
        monkeypatch.delenv("VERIFY_RESPONSES", raising=False)

        llm = create_llm_from_env()

        assert isinstance(llm, GeminiLLM)
        assert llm.model == "gemini-test"
        assert os.environ["VERIFY_RESPONSES"] == "false"

    def test_create_llm_from_env_uses_bedrock(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("BEDROCK_MODEL", "us.test.model")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
        monkeypatch.delenv("VERIFY_RESPONSES", raising=False)

        llm = create_llm_from_env()

        assert isinstance(llm, BedrockLLM)
        assert llm.model == "us.test.model"
        assert os.environ["VERIFY_RESPONSES"] == "false"


class TestSystemPrompt:
    def test_cse_prompt_does_not_invent_hod_alias_email(self):
        prompt = get_system_prompt("computer_science_engineering")
        assert "hod.computer_science_engineering@iitjammu.ac.in" not in prompt
        assert "official department website" in prompt.lower()
