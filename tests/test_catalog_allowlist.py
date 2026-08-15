"""Tests for the catalog-backed tenant allowlist client.

Proves the service authorizes tenants through the benchmark catalog API
(`BENCHMARK_CATALOG_API_URL` + `SERVICE_NAME`) instead of the injected
local allowlist: startup in catalog mode, request shape, fail-closed on
unknown tenants, and catalog-over-legacy precedence.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator, Mapping
from typing import Any

import benchmark_service.auth as auth_module
import httpx
import pytest
from benchmark_service.allowlist import CatalogAllowlistClient
from starlette.testclient import TestClient

from skillsbench_valkyrie.main import app

CATALOG_URL = "https://catalog.test"
SERVICE_NAME = "skillsbench"
DESCOPE_PROJECT_ID = "test-project"
ALLOWED_KEY = "allowed-access-key"
ALLOWED_TENANT = "allowed-tenant"
LEGACY_ONLY_KEY = "legacy-only-access-key"
UNKNOWN_TENANT = "unknown-tenant"
DATASETS = ["default", "with-skills", "extra", "extra-with-skills"]


def _catalog_response_handler(requests_log: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests_log.append(request)
        if request.headers.get("x-descope-api-key") == ALLOWED_KEY:
            return httpx.Response(
                200,
                json={
                    "name": SERVICE_NAME,
                    "datasets": DATASETS,
                    "evaluation_quota": None,
                    "trial_mode": False,
                },
            )
        return httpx.Response(404, json={"detail": "tenant not found"})

    return handler


@pytest.fixture
def requests_log() -> list[httpx.Request]:
    return []


@pytest.fixture(autouse=True)
def catalog_env(monkeypatch: pytest.MonkeyPatch, requests_log: list[httpx.Request]) -> Generator[None, None, None]:
    monkeypatch.setenv("BENCHMARK_CATALOG_API_URL", CATALOG_URL)
    monkeypatch.setenv("SERVICE_NAME", SERVICE_NAME)
    monkeypatch.setenv("DESCOPE_PROJECT_ID", DESCOPE_PROJECT_ID)
    monkeypatch.delenv("AUTH_DISABLED")
    auth_module.clear_allowlist_cache()
    auth_module.clear_auth_cache()

    async def fake_exchange(project_id: str, access_key: str) -> Mapping[str, Any]:
        tenant = (
            ALLOWED_TENANT
            if access_key in {ALLOWED_KEY, LEGACY_ONLY_KEY}
            else UNKNOWN_TENANT
        )
        return {"tenants": {tenant: {}}}

    monkeypatch.setattr(auth_module, "_exchange_descope_access_key", fake_exchange)
    catalog_client = CatalogAllowlistClient(
        CATALOG_URL,
        SERVICE_NAME,
        transport=httpx.MockTransport(_catalog_response_handler(requests_log)),
    )
    monkeypatch.setattr(auth_module, "_catalog_client", catalog_client)
    yield


@pytest.fixture
def client() -> Generator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_app_boots_in_catalog_mode(client: TestClient) -> None:
    """The app starts under catalog configuration and still serves health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response = client.get("/verify-task-ids", params={"task_ids": ["nope"]})
    assert response.status_code == 401


def test_catalog_mode_authorizes_allowed_tenant(
    client: TestClient,
    requests_log: list[httpx.Request],
) -> None:
    """An access key that the catalog allows reaches the protected endpoint."""
    response = client.get(
        "/verify-task-ids",
        params={"task_ids": ["nope"]},
        headers={"x-descope-api-key": ALLOWED_KEY},
    )
    assert response.status_code == 400

    assert requests_log, "expected at least one catalog request"
    request = requests_log[-1]
    assert request.url.path == f"/benchmark-services/{SERVICE_NAME}"
    assert request.headers["x-descope-api-key"] == ALLOWED_KEY


def test_catalog_mode_fails_closed_for_unknown_tenant(client: TestClient) -> None:
    """An access key the catalog does not recognize is rejected."""
    response = client.get(
        "/verify-task-ids",
        params={"task_ids": ["nope"]},
        headers={"x-descope-api-key": "unknown-access-key"},
    )
    assert response.status_code == 401


def test_catalog_mode_ignores_legacy_allowlist(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant listed only in the legacy allowlist is denied in catalog mode."""
    legacy_allowlist = json.dumps({"tenants": {ALLOWED_TENANT: {"datasets": DATASETS}}})
    monkeypatch.setenv("DESCOPE_TENANT_ALLOWLIST_JSON", legacy_allowlist)
    assert auth_module.load_allowlist().tenants == {}

    response = client.get(
        "/verify-task-ids",
        params={"task_ids": ["nope"]},
        headers={"x-descope-api-key": LEGACY_ONLY_KEY},
    )
    assert response.status_code == 401


def test_catalog_client_returns_none_on_network_failure() -> None:
    """Catalog failures must not allow a tenant through."""
    async def broken_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    client_under_test = CatalogAllowlistClient(
        CATALOG_URL,
        SERVICE_NAME,
        transport=httpx.MockTransport(broken_handler),
    )
    config = asyncio.run(client_under_test.get_tenant_config(ALLOWED_KEY, ALLOWED_TENANT))
    assert config is None
