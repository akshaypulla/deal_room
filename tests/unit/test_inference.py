import inference
from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment


def test_inference_prefers_injected_proxy_credentials(monkeypatch):
    monkeypatch.setenv("API_KEY", "proxy-key")
    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-openai")
    monkeypatch.setenv("HF_TOKEN", "fallback-hf")
    inference.get_client.cache_clear()

    api_key, api_base_url = inference.resolve_api_credentials()

    assert api_key == "proxy-key"
    assert api_base_url == "https://proxy.example/v1"
    assert inference.should_use_llm_messages() is True


def test_inference_does_not_force_llm_without_injected_proxy(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-openai")
    monkeypatch.setenv("HF_TOKEN", "fallback-hf")
    monkeypatch.delenv("DEALROOM_ENABLE_LLM_MESSAGES", raising=False)
    inference.get_client.cache_clear()

    api_key, api_base_url = inference.resolve_api_credentials()

    assert api_key == "fallback-openai"
    assert api_base_url == "https://router.huggingface.co/v1"
    assert inference.should_use_llm_messages() is False


def test_explicit_llm_flag_can_enable_local_message_generation(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-openai")
    monkeypatch.setenv("DEALROOM_ENABLE_LLM_MESSAGES", "1")
    inference.get_client.cache_clear()

    assert inference.should_use_llm_messages() is True


def test_inference_uses_openai_client_when_proxy_env_present(monkeypatch):
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1

            class FakeMessage:
                content = '{"message":"Proxy-generated negotiation message."}'

            class FakeChoice:
                message = FakeMessage()

            class FakeResponse:
                choices = [FakeChoice()]

            return FakeResponse()

    class FakeChat:
        def __init__(self, completions):
            self.completions = completions

    class FakeClient:
        def __init__(self):
            self.completions = FakeCompletions()
            self.chat = FakeChat(self.completions)

    fake_client = FakeClient()
    monkeypatch.setenv("API_KEY", "proxy-key")
    monkeypatch.setenv("API_BASE_URL", "https://proxy.example/v1")
    monkeypatch.delenv("DEALROOM_ENABLE_LLM_MESSAGES", raising=False)
    monkeypatch.setattr(inference, "get_client", lambda: fake_client)

    env = DealRoomEnvironment()
    obs = env.reset(task_id="aligned", seed=42)
    action = DealRoomAction(action_type="direct_message", target="finance", target_ids=["finance"])

    updated = inference.action_with_message(
        action,
        obs,
        instruction="Write a concise message that uses the proxy path.",
        fallback_message="Fallback message.",
    )

    assert fake_client.completions.calls == 1
    assert updated.message == "Proxy-generated negotiation message."
