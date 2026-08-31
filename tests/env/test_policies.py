"""The policy layer: one OpenRouter call per observation, and clear failures."""

import pytest

from sphyr import policies


def test_benchmark_names_map_to_openrouter_ids():
    # The stored result directory names are the keys, so a fresh run lands
    # beside the published numbers rather than in a new directory.
    assert policies.resolve_model("claude-opus-4-20250514") == "anthropic/claude-opus-4"
    assert policies.resolve_model("perplexity-sonar") == "perplexity/sonar"


def test_an_openrouter_id_is_passed_through():
    assert (
        policies.resolve_model("anthropic/claude-sonnet-4.5")
        == "anthropic/claude-sonnet-4.5"
    )


def test_a_retired_model_says_so_instead_of_calling_openrouter():
    with pytest.raises(ValueError, match="no longer served"):
        policies.resolve_model("gemini-1.5-pro")


def test_an_unknown_name_is_refused_when_the_policy_is_built():
    with pytest.raises(ValueError, match="Unknown model"):
        policies.get_policy("gpt-9")


def test_a_missing_key_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv(policies.API_KEY_ENV, raising=False)

    with pytest.raises(RuntimeError, match=policies.API_KEY_ENV):
        policies.call_model("perplexity-sonar", "prompt")


class FakeOpenAI:
    """Stands in for the OpenAI SDK client pointed at OpenRouter."""

    def __init__(self, content):
        self.content = content
        self.calls = []

        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message", (), {"content": outer.content}
                                    )()
                                },
                            )()
                        ]
                    },
                )()

        self.chat = type("Chat", (), {"completions": Completions()})()


@pytest.fixture
def fake_client(monkeypatch):
    def install(content):
        client = FakeOpenAI(content)
        monkeypatch.setattr(policies, "_client", lambda name="": client)
        return client

    return install


def test_call_model_sends_the_prompt_to_the_resolved_model(fake_client):
    client = fake_client("1 0\n0 1\n")

    assert policies.call_model("claude-opus-4-20250514", "complete this") == "1 0\n0 1"

    (call,) = client.calls
    assert call["model"] == "anthropic/claude-opus-4"
    assert call["messages"] == [{"role": "user", "content": "complete this"}]
    assert call["max_tokens"] == policies.MAX_TOKENS


def test_an_empty_reply_is_an_error_not_an_empty_grid(fake_client):
    fake_client(None)

    # The runner catches this, records nothing, and the run stays resumable.
    # One attempt: the retry policy is exercised separately, without its waits.
    with pytest.raises(RuntimeError, match="empty reply"):
        policies.call_model("perplexity-sonar", "complete this", max_attempts=1)


def test_a_transient_failure_is_retried(monkeypatch, fake_client):
    """A call that fails once and then succeeds yields the successful reply."""
    monkeypatch.setattr(policies.time, "sleep", lambda _seconds: None)

    client = fake_client("1 0\n0 1")
    create = client.chat.completions.create
    attempts = {"n": 0}

    def flaky(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("connection reset")
        return create(*args, **kwargs)

    client.chat.completions.create = flaky

    assert policies.call_model("perplexity-sonar", "complete this") == "1 0\n0 1"
    assert attempts["n"] == 2


def test_retries_are_exhausted_and_the_last_error_reported(monkeypatch, fake_client):
    """Every attempt failing raises, naming how many were made."""
    monkeypatch.setattr(policies.time, "sleep", lambda _seconds: None)

    client = fake_client("1 0")

    def always_fails(*args, **kwargs):
        raise RuntimeError("connection reset")

    client.chat.completions.create = always_fails

    with pytest.raises(RuntimeError, match="failed 5 times"):
        policies.call_model("perplexity-sonar", "complete this")


def test_the_policy_submits_the_reply_as_the_completed_grid(fake_client):
    fake_client("1 0\n0 1")

    observation = type("Observation", (), {"prompt": "complete this"})()
    action = policies.get_policy("gpt-4.1-2025-04-14")(observation)

    assert action.grid == "1 0\n0 1"
