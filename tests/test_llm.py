import asyncio
import contextlib
import json
import logging
import traceback
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from product_card.config import Settings
from product_card.llm import (
    LLMBadOutput,
    LLMError,
    LLMResult,
    OpenRouterClient,
)
from product_card.prompts import DRAFT_JSON_SCHEMA

from .fakes import make_settings

API_KEY = "sk-or" + "-v1-" + "a1b2c3d4" * 6
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
SENSITIVE = "UPSTREAM-SENSITIVE-TEXT"

Handler = Callable[[httpx.Request], httpx.Response]


def ok_body(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "anthropic/claude-haiku-5.5",
        "choices": [{"finish_reason": "stop", "message": {"content": '{"a": 1}'}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22, "cost": 0.0005},
    }
    body.update(overrides)
    return body


def ok_response(**overrides: object) -> httpx.Response:
    return httpx.Response(200, json=ok_body(**overrides))


class Recorder:
    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def last_body(self) -> dict[str, object]:
        return json.loads(self.requests[-1].content)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("product_card.llm.asyncio.sleep", fake_sleep)
    return recorded


def call(
    handler: Handler,
    *,
    models: list[str] | None = None,
    **settings: object,
) -> LLMResult:
    async def scenario() -> LLMResult:
        client = OpenRouterClient(
            make_settings(openrouter_api_key=API_KEY, **settings),  # type: ignore[arg-type]
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.complete(MESSAGES, models=models)
        finally:
            await client.aclose()

    return asyncio.run(scenario())


def failure(handler: Handler, **settings: object) -> LLMError:
    with pytest.raises(LLMError) as caught:
        call(handler, **settings)
    return caught.value


def test_successful_response_is_parsed(sleeps: list[float]) -> None:
    result = call(Recorder(ok_response()))

    assert result == LLMResult(
        content='{"a": 1}',
        model="anthropic/claude-haiku-5.5",
        prompt_tokens=11,
        completion_tokens=22,
        cost_usd=0.0005,
    )
    assert sleeps == []


def test_missing_usage_gives_zero_tokens_and_unknown_cost() -> None:
    result = call(Recorder(ok_response(usage=None)))

    assert (result.prompt_tokens, result.completion_tokens, result.cost_usd) == (0, 0, None)


def test_request_goes_to_chat_completions_with_authorization() -> None:
    recorder = Recorder(ok_response())

    call(recorder)

    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == f"Bearer {API_KEY}"


def test_request_body_routes_models_and_requires_parameters() -> None:
    recorder = Recorder(ok_response())
    settings = make_settings()

    call(recorder)

    body = recorder.last_body
    assert body["models"] == settings.models
    assert body["messages"] == MESSAGES
    assert body["max_tokens"] == settings.llm_max_tokens
    provider = body["provider"]
    assert provider["require_parameters"] is True  # type: ignore[index]
    assert provider["data_collection"] == "deny"  # type: ignore[index]
    assert provider["max_price"] == {"prompt": 5.0, "completion": 20.0}  # type: ignore[index]


def test_request_body_uses_strict_json_schema_and_no_temperature() -> None:
    recorder = Recorder(ok_response())

    call(recorder)

    body = recorder.last_body
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"  # type: ignore[index]
    assert response_format["json_schema"]["strict"] is True  # type: ignore[index]
    assert response_format["json_schema"]["schema"] == DRAFT_JSON_SCHEMA  # type: ignore[index]
    assert "temperature" not in body


def test_explicit_models_override_the_default_chain() -> None:
    recorder = Recorder(ok_response())

    call(recorder, models=["vendor/strong"])

    assert recorder.last_body["models"] == ["vendor/strong"]


def test_data_collection_and_price_ceiling_follow_settings() -> None:
    recorder = Recorder(ok_response())

    call(
        recorder,
        llm_data_collection="allow",
        llm_max_price_prompt=1.5,
        llm_max_price_completion=3.0,
    )

    provider = recorder.last_body["provider"]
    assert provider["data_collection"] == "allow"  # type: ignore[index]
    assert provider["max_price"] == {"prompt": 1.5, "completion": 3.0}  # type: ignore[index]


def test_zdr_is_absent_by_default() -> None:
    recorder = Recorder(ok_response())

    call(recorder)

    assert "zdr" not in recorder.last_body["provider"]  # type: ignore[operator]


def test_zdr_is_sent_when_enabled() -> None:
    recorder = Recorder(ok_response())

    call(recorder, llm_zdr=True)

    assert recorder.last_body["provider"]["zdr"] is True  # type: ignore[index]


@pytest.mark.parametrize(
    ("status", "code", "http_status"),
    [
        (400, "llm_bad_request", 502),
        (401, "llm_auth", 502),
        (402, "llm_credits", 502),
        (403, "llm_moderated", 502),
        (404, "llm_no_route", 502),
        (408, "llm_timeout", 504),
        (429, "llm_rate_limited", 429),
        (500, "llm_unavailable", 503),
        (502, "llm_unavailable", 503),
        (503, "llm_unavailable", 503),
    ],
)
def test_http_status_is_mapped_to_error_code(
    status: int, code: str, http_status: int, sleeps: list[float]
) -> None:
    error = failure(Recorder(httpx.Response(status, text="x")))

    assert (error.code, error.http_status) == (code, http_status)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
def test_retryable_status_is_retried_up_to_the_limit(status: int, sleeps: list[float]) -> None:
    recorder = Recorder(httpx.Response(status, text="x"))

    failure(recorder, llm_http_retries=2)

    assert len(recorder.requests) == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_client_errors_are_not_retried(status: int, sleeps: list[float]) -> None:
    recorder = Recorder(httpx.Response(status, text="x"))

    failure(recorder, llm_http_retries=3)

    assert len(recorder.requests) == 1
    assert sleeps == []


def test_zero_retries_means_a_single_request(sleeps: list[float]) -> None:
    recorder = Recorder(httpx.Response(503, text="x"))

    failure(recorder, llm_http_retries=0)

    assert len(recorder.requests) == 1


def test_success_after_transient_failures(sleeps: list[float]) -> None:
    recorder = Recorder(httpx.Response(503), httpx.Response(429), ok_response())

    result = call(recorder, llm_http_retries=2)

    assert result.content == '{"a": 1}'
    assert len(recorder.requests) == 3


def test_backoff_grows_between_retries(sleeps: list[float]) -> None:
    failure(Recorder(httpx.Response(503)), llm_http_retries=3)

    assert [int(delay) for delay in sleeps] == [1, 2, 4]


def test_retry_after_header_sets_the_delay(sleeps: list[float]) -> None:
    recorder = Recorder(httpx.Response(429, headers={"Retry-After": "3"}), ok_response())

    call(recorder)

    assert sleeps == [3.0]


def test_retry_after_is_capped(sleeps: list[float]) -> None:
    recorder = Recorder(httpx.Response(429, headers={"Retry-After": "3600"}), ok_response())

    call(recorder)

    assert sleeps == [20.0]


@pytest.mark.parametrize("value", ["soon", "Wed, 21 Oct 2026 07:28:00 GMT", ""])
def test_unparseable_retry_after_falls_back_to_backoff(value: str, sleeps: list[float]) -> None:
    recorder = Recorder(httpx.Response(503, headers={"Retry-After": value}), ok_response())

    call(recorder)

    assert 1 <= sleeps[0] <= 1.25


def test_timeout_is_retried_and_mapped(sleeps: list[float]) -> None:
    recorder = Recorder(httpx.ReadTimeout("slow"))

    error = failure(recorder, llm_http_retries=2)

    assert (error.code, error.http_status) == ("llm_timeout", 504)
    assert len(recorder.requests) == 3


def test_connection_error_is_retried_and_mapped(sleeps: list[float]) -> None:
    recorder = Recorder(httpx.ConnectError("refused"))

    error = failure(recorder, llm_http_retries=1)

    assert (error.code, error.http_status) == ("llm_unavailable", 503)
    assert len(recorder.requests) == 2


def test_timeout_then_success(sleeps: list[float]) -> None:
    result = call(Recorder(httpx.ConnectTimeout("slow"), ok_response()))

    assert result.content == '{"a": 1}'


def test_error_inside_200_body_is_mapped(sleeps: list[float]) -> None:
    body = {"error": {"code": 402, "message": SENSITIVE}}
    recorder = Recorder(httpx.Response(200, json=body))

    error = failure(recorder)

    assert error.code == "llm_credits"
    assert len(recorder.requests) == 1


def test_retryable_error_inside_200_body_is_retried(sleeps: list[float]) -> None:
    body = {"error": {"code": 429, "message": SENSITIVE}}
    recorder = Recorder(httpx.Response(200, json=body))

    error = failure(recorder, llm_http_retries=2)

    assert (error.code, error.http_status) == ("llm_rate_limited", 429)
    assert len(recorder.requests) == 3


def test_error_in_choice_inside_200_body_is_mapped(sleeps: list[float]) -> None:
    body = ok_body(choices=[{"error": {"code": 403, "message": SENSITIVE}}])

    error = failure(Recorder(httpx.Response(200, json=body)))

    assert error.code == "llm_moderated"


def test_error_without_numeric_code_is_treated_as_unavailable(sleeps: list[float]) -> None:
    body = {"error": {"message": SENSITIVE}}

    error = failure(Recorder(httpx.Response(200, json=body)), llm_http_retries=0)

    assert error.code == "llm_unavailable"


def test_finish_reason_length_is_bad_output_and_not_retried(sleeps: list[float]) -> None:
    body = ok_body(choices=[{"finish_reason": "length", "message": {"content": '{"a"'}}])
    recorder = Recorder(httpx.Response(200, json=body))

    with pytest.raises(LLMBadOutput):
        call(recorder)

    assert len(recorder.requests) == 1


@pytest.mark.parametrize("content", ["", "   ", None, 5, ["x"]])
def test_empty_or_non_text_content_is_bad_output(content: object) -> None:
    body = ok_body(choices=[{"finish_reason": "stop", "message": {"content": content}}])

    with pytest.raises(LLMBadOutput):
        call(Recorder(httpx.Response(200, json=body)))


@pytest.mark.parametrize("choices", [[], None, "x", [None], [{}], [{"message": {}}]])
def test_missing_choices_are_bad_output(choices: object) -> None:
    with pytest.raises(LLMBadOutput):
        call(Recorder(httpx.Response(200, json=ok_body(choices=choices))))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, text=""),
        httpx.Response(200, json=["not", "an", "object"]),
    ],
)
def test_non_json_or_non_object_body_is_bad_output(response: httpx.Response) -> None:
    with pytest.raises(LLMBadOutput):
        call(Recorder(response))


def test_bad_output_error_is_an_llm_error() -> None:
    assert issubclass(LLMBadOutput, LLMError)
    assert LLMBadOutput().code == "llm_bad_output"


def test_redirect_is_not_followed(sleeps: list[float]) -> None:
    redirect = httpx.Response(302, headers={"Location": "https://evil.example/steal"})
    recorder = Recorder(redirect)

    error = failure(recorder)

    assert len(recorder.requests) == 1
    assert {request.url.host for request in recorder.requests} == {"openrouter.ai"}
    assert error.code == "llm_unavailable"


def exception_text(error: BaseException) -> str:
    return "".join(traceback.format_exception(error)) + repr(error) + str(error)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, text=f"bad request {SENSITIVE} Bearer {API_KEY}"),
        httpx.Response(401, json={"error": {"message": f"{SENSITIVE} {API_KEY}"}}),
        httpx.Response(402, text=SENSITIVE),
        httpx.Response(500, text=SENSITIVE),
        httpx.Response(200, json={"error": {"code": 400, "message": f"{SENSITIVE} {API_KEY}"}}),
        httpx.Response(200, text=f"{SENSITIVE} {API_KEY}"),
    ],
)
def test_upstream_body_and_key_never_reach_the_error(
    response: httpx.Response, sleeps: list[float]
) -> None:
    with pytest.raises(LLMError) as caught:
        call(Recorder(response), llm_http_retries=1)

    text = exception_text(caught.value)
    assert SENSITIVE not in text
    assert API_KEY not in text


@pytest.mark.parametrize(
    "exception",
    [
        httpx.ConnectError(f"cannot connect with {API_KEY}"),
        httpx.ReadTimeout(f"timeout {API_KEY}"),
        httpx.RemoteProtocolError(f"protocol {API_KEY} {SENSITIVE}"),
    ],
)
def test_transport_exception_text_never_reaches_the_error(
    exception: Exception, sleeps: list[float]
) -> None:
    with pytest.raises(LLMError) as caught:
        call(Recorder(exception), llm_http_retries=1)

    text = exception_text(caught.value)
    assert API_KEY not in text
    assert SENSITIVE not in text


def test_api_key_never_reaches_the_logs(
    caplog: pytest.LogCaptureFixture, sleeps: list[float]
) -> None:
    caplog.set_level(logging.DEBUG)
    scenarios = [
        Recorder(httpx.ConnectError(f"boom {API_KEY}")),
        Recorder(httpx.Response(401, text=f"{API_KEY} {SENSITIVE}")),
        Recorder(httpx.Response(503, text=SENSITIVE)),
        Recorder(ok_response()),
    ]

    for scenario in scenarios:
        with contextlib.suppress(LLMError):
            call(scenario, llm_http_retries=1)

    assert API_KEY not in caplog.text
    assert SENSITIVE not in caplog.text


def test_api_key_is_not_in_the_request_body() -> None:
    recorder = Recorder(ok_response())

    call(recorder)

    assert API_KEY not in recorder.requests[0].content.decode()


@pytest.mark.parametrize(
    "url",
    [
        "http://openrouter.ai/api/v1",
        "http://example.com",
        "http://localhost.evil.example/v1",
        "http://localhost@evil.example/v1",
        "http://evil.example#@localhost",
        "ftp://localhost/v1",
        "file:///etc/passwd",
        "https://",
        "openrouter.ai/api/v1",
        "",
    ],
)
def test_settings_reject_unsafe_base_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(openrouter_base_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://localhost:8080/v1",
        "http://127.0.0.1:9000/api",
        "http://[::1]:9000/api",
        "https://openrouter.ai/api/v1",
        "https://proxy.example.com/v1",
    ],
)
def test_settings_accept_https_and_loopback_http_base_urls(url: str) -> None:
    assert make_settings(openrouter_base_url=url).openrouter_base_url == url


def test_base_url_trailing_slash_is_removed() -> None:
    assert make_settings(openrouter_base_url="https://x.example/v1/").openrouter_base_url == (
        "https://x.example/v1"
    )


def test_settings_repr_and_str_hide_the_api_key() -> None:
    settings = make_settings(openrouter_api_key=API_KEY)

    assert API_KEY not in repr(settings)
    assert API_KEY not in str(settings)
    assert API_KEY not in repr(settings.openrouter_api_key)
    assert settings.openrouter_api_key.get_secret_value() == API_KEY  # type: ignore[union-attr]


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_api_key_counts_as_missing(value: str) -> None:
    assert make_settings(openrouter_api_key=value).openrouter_api_key is None


def test_api_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)

    assert make_settings().openrouter_api_key.get_secret_value() == API_KEY  # type: ignore[union-attr]


def test_developer_env_file_is_ignored_by_test_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(f"OPENROUTER_API_KEY={API_KEY}\n", "utf-8")
    monkeypatch.chdir(tmp_path)

    assert make_settings().openrouter_api_key is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_models", " , "),
        ("llm_max_attempts", 0),
        ("llm_max_attempts", 5),
        ("llm_http_retries", 6),
        ("llm_timeout_seconds", 0),
        ("llm_max_price_prompt", 0),
        ("max_concurrent_generations", 0),
        ("llm_data_collection", "maybe"),
    ],
)
def test_settings_reject_out_of_range_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{field: value})


def test_model_lists_are_split_and_trimmed() -> None:
    settings = make_settings(llm_models=" a/one , b/two ,", llm_escalation_models="")

    assert settings.models == ["a/one", "b/two"]
    assert settings.escalation_models == ["a/one", "b/two"]


@pytest.mark.parametrize("variable", ["llm_models", "llm_escalation_models"])
def test_settings_reject_a_model_chain_longer_than_openrouter_allows(variable):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{variable: "a/1,a/2,a/3,a/4"})


def test_reasoning_budget_is_limited_in_the_request():
    settings = make_settings(openrouter_api_key="key")
    body = OpenRouterClient(settings)._build_body([], ["a/1"])
    assert body["reasoning"] == {"effort": "low", "exclude": True}


def test_prompt_mode_opens_up_models_without_json_schema():
    settings = Settings(_env_file=None, openrouter_api_key="key", llm_output_mode="prompt")
    messages = [{"role": "system", "content": "правила"}, {"role": "user", "content": "товар"}]
    body = OpenRouterClient(settings)._build_body(messages, ["a/1"])
    assert "response_format" not in body
    assert "plugins" not in body
    assert "require_parameters" not in body["provider"]
    assert "seo_description" in body["messages"][0]["content"]
    assert messages[0]["content"] == "правила"


def test_rate_limit_without_retry_after_waits_longer_than_a_plain_failure():
    plain = OpenRouterClient._retry_delay(1, None)
    limited = OpenRouterClient._retry_delay(1, None, rate_limited=True)
    assert limited > plain + 1


def test_local_server_gets_one_model_and_no_routing_fields():
    settings = make_settings(
        openrouter_api_key="key",
        openrouter_base_url="http://localhost:11434/v1",
        llm_output_mode="prompt",
        llm_reasoning_effort="none",
    )
    messages = [{"role": "system", "content": "правила"}, {"role": "user", "content": "товар"}]
    body = OpenRouterClient(settings)._build_body(messages, ["local/model", "local/other"])
    assert body["model"] == "local/model"
    assert body["reasoning_effort"] == "none"
    assert not {"models", "provider", "plugins", "reasoning", "response_format"} & set(body)
    assert "seo_description" in body["messages"][0]["content"]


def test_openrouter_body_keeps_routing_fields():
    settings = make_settings(openrouter_api_key="key")
    body = OpenRouterClient(settings)._build_body([], ["a/1"])
    assert body["models"] == ["a/1"]
    assert "model" not in body
    assert body["provider"]["require_parameters"] is True
