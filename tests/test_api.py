import json
import logging
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from product_card.api import create_app
from product_card.llm import LLMError

from .fakes import ScriptedLLM, dm02_raw, draft_payload, invented_range_payload, make_settings

JSON = {"Content-Type": "application/json"}
SECRET = "SECRET-INPUT-MARKER"


def client_for(
    llm: ScriptedLLM | None = None, *, raise_server_exceptions: bool = True, **settings: object
) -> TestClient:
    settings.setdefault("rate_limit_per_minute", 1000)
    app = create_app(make_settings(**settings), llm)  # type: ignore[arg-type]
    return TestClient(
        app, base_url="http://localhost", raise_server_exceptions=raise_server_exceptions
    )


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM([draft_payload()])


@pytest.fixture
def client(llm: ScriptedLLM) -> Iterator[TestClient]:
    with client_for(llm) as test_client:
        yield test_client


def post_product(client: TestClient, raw: object | None = None, **kwargs: object) -> httpx.Response:
    body = json.dumps(dm02_raw() if raw is None else raw, ensure_ascii=False)
    headers = {**JSON, **kwargs.pop("headers", {})}  # type: ignore[arg-type]
    return client.post("/api/generate", content=body.encode(), headers=headers, **kwargs)  # type: ignore[arg-type]


def assert_security_headers(response: httpx.Response) -> None:
    headers = response.headers
    assert "default-src 'none'" in headers["content-security-policy"]
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"


def test_successful_generation_returns_card_and_exports(client: TestClient) -> None:
    response = post_product(client)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"card", "exports"}
    assert body["card"]["status"] == "ready"
    assert body["card"]["seo_title"] == draft_payload()["seo_title"]
    assert body["card"]["meta"]["cached"] is False
    assert body["exports"]["markdown"].startswith("# ")
    assert "Манометр показывающий DM 02" in body["exports"]["text"]


def test_second_identical_request_is_served_from_cache(
    client: TestClient, llm: ScriptedLLM
) -> None:
    post_product(client)
    second = post_product(client)

    assert second.json()["card"]["meta"]["cached"] is True
    assert len(llm.calls) == 1


def test_fresh_query_parameter_bypasses_the_cache(client: TestClient, llm: ScriptedLLM) -> None:
    post_product(client)
    fresh = post_product(client, params={"fresh": "true"})

    assert fresh.json()["card"]["meta"]["cached"] is False
    assert len(llm.calls) == 2


def test_unknown_fields_are_reported_in_the_card(client: TestClient) -> None:
    response = post_product(client, dm02_raw(price=100))

    missing = response.json()["card"]["missing_fields"]
    assert "price" in {item["field"] for item in missing}


def test_insufficient_data_is_a_200_with_that_status(client: TestClient, llm: ScriptedLLM) -> None:
    response = post_product(client, {"name": ""})

    assert response.status_code == 200
    assert response.json()["card"]["status"] == "insufficient_data"
    assert llm.calls == []


def test_invalid_json_gives_400_with_position_and_no_echo(client: TestClient) -> None:
    body = f'{{"name": "{SECRET}",\n "specs" oops}}'

    response = client.post("/api/generate", content=body.encode(), headers=JSON)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_json"
    assert "строка 2" in error["message"]
    assert SECRET not in response.text


def test_empty_body_gives_400(client: TestClient) -> None:
    response = client.post("/api/generate", content=b"", headers=JSON)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_invalid_utf8_gives_400(client: TestClient) -> None:
    response = client.post("/api/generate", content=b'{"name": "\xff\xfe"}', headers=JSON)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


def test_nan_gives_400(client: TestClient) -> None:
    response = client.post("/api/generate", content=b'{"name": NaN}', headers=JSON)

    assert response.status_code == 400


def test_invalid_fields_give_422_with_details(client: TestClient) -> None:
    response = post_product(client, {"name": [SECRET], "specs": []})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_product"
    assert [item["field"] for item in error["details"]] == ["name", "specs"]
    assert SECRET not in response.text


def test_non_object_root_gives_422(client: TestClient) -> None:
    response = post_product(client, [dm02_raw()])

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_product"


def test_invalid_query_parameter_gives_422_without_echo(client: TestClient) -> None:
    response = post_product(client, params={"fresh": SECRET})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert SECRET not in response.text


def test_oversize_body_with_content_length_gives_413(client: TestClient, llm: ScriptedLLM) -> None:
    response = post_product(client, dm02_raw(description="а" * 70_000))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_too_large"
    assert llm.calls == []


def test_oversize_chunked_body_without_content_length_gives_413(
    client: TestClient, llm: ScriptedLLM
) -> None:
    def chunks() -> Iterator[bytes]:
        yield b'{"description": "'
        for _ in range(80):
            yield b"a" * 1000
        yield b'"}'

    request = client.build_request("POST", "/api/generate", content=chunks(), headers=JSON)
    assert "content-length" not in request.headers

    response = client.send(request)

    assert response.status_code == 413
    assert llm.calls == []


def test_body_at_the_limit_is_accepted(llm: ScriptedLLM) -> None:
    with client_for(llm, max_body_bytes=2_000) as small:
        body = json.dumps(dm02_raw(), ensure_ascii=False).encode()
        padded = body[:-1] + b" " * (2_000 - len(body)) + b"}"
        assert len(padded) == 2_000

        response = small.post("/api/generate", content=padded, headers=JSON)

    assert response.status_code == 200


def test_body_over_the_limit_by_one_byte_gives_413(llm: ScriptedLLM) -> None:
    with client_for(llm, max_body_bytes=2_000) as small:
        body = json.dumps(dm02_raw(), ensure_ascii=False).encode()
        padded = body[:-1] + b" " * (2_001 - len(body)) + b"}"
        assert len(padded) == 2_001

        response = small.post("/api/generate", content=padded, headers=JSON)

    assert response.status_code == 413


@pytest.mark.parametrize(
    "content_type",
    [
        "text/plain",
        "application/x-www-form-urlencoded",
        "multipart/form-data; boundary=x",
        "application/xml",
        "",
    ],
)
def test_non_json_content_type_gives_415(
    client: TestClient, llm: ScriptedLLM, content_type: str
) -> None:
    headers = {"Content-Type": content_type} if content_type else {}

    response = client.post("/api/generate", content=b'{"name": "x"}', headers=headers)

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert llm.calls == []


def test_json_content_type_with_charset_is_accepted(client: TestClient) -> None:
    response = post_product(client, headers={"Content-Type": "Application/JSON; charset=utf-8"})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "https://localhost.evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
        {"Origin": "http://evil.example", "Sec-Fetch-Site": "cross-site"},
    ],
)
def test_foreign_origin_gives_403_and_makes_no_llm_call(
    client: TestClient, llm: ScriptedLLM, headers: dict[str, str]
) -> None:
    response = post_product(client, headers=headers)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden_origin"
    assert llm.calls == []


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://localhost"},
        {"Origin": "http://127.0.0.1:8000", "Host": "127.0.0.1:8000"},
        {"Sec-Fetch-Site": "same-origin"},
        {"Sec-Fetch-Site": "none"},
    ],
)
def test_same_origin_requests_are_accepted(client: TestClient, headers: dict[str, str]) -> None:
    assert post_product(client, headers=headers).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "localhost.evil.example", "10.0.0.5"])
def test_foreign_host_gives_400(client: TestClient, llm: ScriptedLLM, host: str) -> None:
    response = client.get("/api/health", headers={"Host": host})

    assert response.status_code == 400
    assert llm.calls == []


def test_foreign_host_cannot_reach_generation(client: TestClient, llm: ScriptedLLM) -> None:
    response = post_product(client, headers={"Host": "rebind.evil.example"})

    assert response.status_code == 400
    assert llm.calls == []


def test_rate_limit_gives_429_with_retry_after(llm: ScriptedLLM) -> None:
    with client_for(llm, rate_limit_per_minute=2) as limited:
        statuses = [post_product(limited).status_code for _ in range(2)]
        blocked = post_product(limited)

    assert statuses == [200, 200]
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert 1 <= int(blocked.headers["retry-after"]) <= 61


def test_rate_limit_does_not_apply_to_health(llm: ScriptedLLM) -> None:
    with client_for(llm, rate_limit_per_minute=1) as limited:
        statuses = {limited.get("/api/health").status_code for _ in range(5)}

    assert statuses == {200}


def test_missing_api_key_gives_503_llm_not_configured() -> None:
    with client_for(None) as unconfigured:
        response = post_product(unconfigured)
        health = unconfigured.get("/api/health").json()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_not_configured"
    assert health["llm_configured"] is False


def test_missing_api_key_still_reports_bad_input_first() -> None:
    with client_for(None) as unconfigured:
        response = unconfigured.post("/api/generate", content=b"{", headers=JSON)

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMError("llm_credits", "Нет средств.", 502), 502, "llm_credits"),
        (LLMError("llm_rate_limited", "Лимит.", 429), 429, "llm_rate_limited"),
        (LLMError("llm_timeout", "Таймаут.", 504), 504, "llm_timeout"),
    ],
)
def test_llm_errors_are_mapped_to_their_status(error: LLMError, status: int, code: str) -> None:
    with client_for(ScriptedLLM([error])) as failing:
        response = post_product(failing)

    assert response.status_code == status
    assert response.json()["error"] == {"code": code, "message": error.message, "details": []}


def test_transport_error_from_llm_gives_503_without_internals() -> None:
    llm = ScriptedLLM([httpx.ConnectError("connect failed to internal-host-7")])
    with client_for(llm) as failing:
        response = post_product(failing)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "llm_unavailable"
    assert "internal-host-7" not in response.text


def test_needs_review_card_is_returned_with_issues() -> None:
    with client_for(ScriptedLLM([invented_range_payload()])) as reviewing:
        response = post_product(reviewing)

    card = response.json()["card"]
    assert response.status_code == 200
    assert card["status"] == "needs_review"
    assert any(issue["code"] == "unsupported_number" for issue in card["issues"])


def test_unexpected_exception_gives_generic_500(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    llm = ScriptedLLM([RuntimeError("internal-secret-detail /srv/app/path")])
    with client_for(llm, raise_server_exceptions=False) as broken:
        response = post_product(broken)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "internal-secret-detail" not in response.text
    assert "Traceback" not in response.text
    assert "internal-secret-detail" not in caplog.text


@pytest.mark.parametrize("path", ["/", "/api/health", "/api/examples"])
def test_security_headers_on_successful_responses(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert_security_headers(response)


@pytest.mark.parametrize("path", ["/api/health", "/api/examples"])
def test_api_responses_are_not_cacheable(client: TestClient, path: str) -> None:
    assert client.get(path).headers["cache-control"] == "no-store"


def test_security_headers_on_handled_error_responses(client: TestClient) -> None:
    responses = [
        client.post("/api/generate", content=b"{", headers=JSON),
        client.post("/api/generate", content=b"x", headers={"Content-Type": "text/plain"}),
        post_product(client, [1]),
        post_product(client, headers={"Origin": "http://evil.example"}),
        client.get("/api/generate"),
        client.get("/no-such-page"),
    ]

    for response in responses:
        assert response.status_code >= 400
        assert_security_headers(response)


def test_api_error_responses_are_not_cacheable(client: TestClient) -> None:
    response = client.post("/api/generate", content=b"{", headers=JSON)

    assert response.headers["cache-control"] == "no-store"


def test_security_headers_on_unexpected_error_response() -> None:
    llm = ScriptedLLM([RuntimeError("boom")])
    with client_for(llm, raise_server_exceptions=False) as broken:
        response = post_product(broken)

    assert response.status_code == 500
    assert_security_headers(response)
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_interactive_docs_and_schema_are_not_exposed(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_health_reports_configuration_without_secrets(client: TestClient) -> None:
    body = client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["llm_configured"] is True
    assert body["models"] == make_settings().models
    assert "key" not in json.dumps(body).lower()


def test_examples_endpoint_lists_the_bundled_examples(client: TestClient) -> None:
    examples = client.get("/api/examples").json()["examples"]

    assert "manometer_dm02" in {item["id"] for item in examples}


def test_get_on_generate_endpoint_gives_json_error(client: TestClient) -> None:
    response = client.get("/api/generate")

    assert response.status_code in {404, 405}
    assert response.json()["error"]["code"] == "http_error"


def test_error_responses_never_echo_the_request_body(client: TestClient) -> None:
    payloads = [
        (JSON, f'{{"name": "{SECRET}" oops}}'),
        (JSON, json.dumps({"name": [SECRET]})),
        (JSON, json.dumps([SECRET])),
        ({"Content-Type": "text/plain"}, SECRET),
        (JSON, json.dumps({"name": SECRET, "description": "а" * 70_000})),
    ]

    for headers, body in payloads:
        response = client.post("/api/generate", content=body.encode(), headers=headers)
        assert response.status_code >= 400
        assert SECRET not in response.text


def test_security_headers_on_rejected_host():
    with TestClient(
        create_app(make_settings(), ScriptedLLM([draft_payload()])), base_url="http://evil.example"
    ) as client:
        response = client.get("/api/health")
    assert response.status_code == 400
    assert "Content-Security-Policy" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_insufficient_data_is_reported_without_a_key():
    settings = make_settings(openrouter_api_key=None)
    with TestClient(create_app(settings), base_url="http://localhost") as client:
        response = client.post("/api/generate", json={"name": ""})
    assert response.status_code == 200
    assert response.json()["card"]["status"] == "insufficient_data"


def test_another_port_on_localhost_is_a_foreign_origin():
    llm = ScriptedLLM([draft_payload()])
    with TestClient(create_app(make_settings(), llm), base_url="http://localhost:8000") as client:
        response = client.post(
            "/api/generate", json=dm02_raw(), headers={"Origin": "http://localhost:3000"}
        )
    assert response.status_code == 403
