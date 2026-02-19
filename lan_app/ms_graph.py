"""Microsoft Graph delegated auth via Device Code Flow + token cache."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import msal
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import AppSettings

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
_DEVICE_FLOW_TTL_SECONDS = 20 * 60
_MAX_PENDING_DEVICE_FLOWS = 1


class GraphAuthError(RuntimeError):
    """Base class for Microsoft Graph auth/connection failures."""


class GraphNotConfiguredError(GraphAuthError):
    """Raised when Microsoft auth settings are missing."""


class GraphNeedsReconnectError(GraphAuthError):
    """Raised when cached auth cannot be used and reconnect is required."""


class GraphRequestError(GraphAuthError):
    """Raised when Graph responds with a non-retriable request error."""


class GraphDeviceFlowLimitError(GraphAuthError):
    """Raised when too many device-code sessions are currently pending."""


class _GraphTransientError(RuntimeError):
    """Raised for transient Graph/transport failures to trigger retries."""


@dataclass
class _DeviceFlowSession:
    flow: dict[str, Any]
    status: str = "pending"
    error: str | None = None
    account_display_name: str | None = None
    tenant_id: str | None = None
    granted_scopes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


_DEVICE_FLOW_LOCK = threading.Lock()
_DEVICE_FLOW_SESSIONS: dict[str, _DeviceFlowSession] = {}


def _scopes_from_token_result(result: dict[str, Any] | None) -> list[str]:
    if not result:
        return []
    scope = str(result.get("scope", "")).strip()
    if not scope:
        return []
    return [item for item in scope.split(" ") if item]


def _load_token_cache(path: Path) -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if not path.exists():
        return cache
    try:
        cache.deserialize(path.read_text(encoding="utf-8"))
    except Exception:
        # Corrupted cache should not crash the app; reconnect will recreate it.
        return msal.SerializableTokenCache()
    return cache


def _persist_token_cache(cache: msal.SerializableTokenCache, path: Path) -> None:
    if not cache.has_state_changed:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cache.serialize(), encoding="utf-8")


def _token_display_name(result: dict[str, Any]) -> str | None:
    claims = result.get("id_token_claims") or {}
    if not isinstance(claims, dict):
        return None
    return claims.get("name") or claims.get("preferred_username")


def _token_tenant_id(result: dict[str, Any]) -> str | None:
    claims = result.get("id_token_claims") or {}
    if not isinstance(claims, dict):
        return None
    tid = claims.get("tid")
    return str(tid) if tid else None


def _prune_sessions_locked(now: float) -> None:
    stale: list[str] = []
    for session_id, session in _DEVICE_FLOW_SESSIONS.items():
        age = now - session.created_at
        if age > _DEVICE_FLOW_TTL_SECONDS:
            stale.append(session_id)
            continue
        if session.finished_at is None:
            continue
        if now - session.finished_at > _DEVICE_FLOW_TTL_SECONDS:
            stale.append(session_id)
    for session_id in stale:
        _DEVICE_FLOW_SESSIONS.pop(session_id, None)


def _session_payload(session_id: str, session: _DeviceFlowSession) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": session.status,
        "error": session.error,
        "user_code": session.flow.get("user_code"),
        "verification_uri": session.flow.get("verification_uri"),
        "verification_uri_complete": session.flow.get("verification_uri_complete"),
        "message": session.flow.get("message"),
        "expires_in": session.flow.get("expires_in"),
        "account_display_name": session.account_display_name,
        "tenant_id": session.tenant_id,
        "granted_scopes": session.granted_scopes,
    }


def _pending_session_ids_locked() -> list[str]:
    return [sid for sid, session in _DEVICE_FLOW_SESSIONS.items() if session.status == "pending"]


def _first_pending_session_locked() -> tuple[str, _DeviceFlowSession] | None:
    for session_id in _pending_session_ids_locked():
        session = _DEVICE_FLOW_SESSIONS.get(session_id)
        if session is not None:
            return session_id, session
    return None


class MicrosoftGraphClient:
    """Thin wrapper around MSAL + Microsoft Graph REST calls."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings()
        self.tenant_id = (self.settings.ms_tenant_id or "").strip()
        self.client_id = (self.settings.ms_client_id or "").strip()
        self.scopes = self.settings.ms_scopes_list
        self.cache_path = self.settings.msal_cache_path
        self._cache = _load_token_cache(self.cache_path)
        self._last_token_result: dict[str, Any] | None = None
        self._app: msal.PublicClientApplication | None = None
        if self.is_configured:
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
            self._app = msal.PublicClientApplication(
                client_id=self.client_id,
                authority=authority,
                token_cache=self._cache,
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.tenant_id and self.client_id and self.scopes)

    def _require_app(self) -> msal.PublicClientApplication:
        if self._app is None:
            missing: list[str] = []
            if not self.tenant_id:
                missing.append("MS_TENANT_ID")
            if not self.client_id:
                missing.append("MS_CLIENT_ID")
            if not self.scopes:
                missing.append("MS_SCOPES")
            missing_str = ", ".join(missing) if missing else "settings"
            raise GraphNotConfiguredError(
                f"Microsoft Graph auth is not configured ({missing_str})."
            )
        return self._app

    def clear_cache(self) -> None:
        try:
            self.cache_path.unlink()
        except FileNotFoundError:
            pass

    def get_cached_account(self) -> dict[str, Any] | None:
        app = self._require_app()
        accounts = app.get_accounts()
        if not accounts:
            return None
        account = accounts[0]
        return {
            "display_name": account.get("name") or account.get("username"),
            "username": account.get("username"),
            "tenant_id": account.get("tenant_id"),
        }

    def initiate_device_flow(self) -> dict[str, Any]:
        app = self._require_app()
        flow = app.initiate_device_flow(scopes=self.scopes)
        if "user_code" not in flow:
            detail = flow.get("error_description") or flow.get("error") or str(flow)
            raise GraphAuthError(f"Failed to start device code flow: {detail}")
        return flow

    def acquire_token_by_device_flow(self, flow: dict[str, Any]) -> dict[str, Any]:
        app = self._require_app()
        result = app.acquire_token_by_device_flow(flow)
        _persist_token_cache(self._cache, self.cache_path)
        if "access_token" not in result:
            detail = result.get("error_description") or result.get("error") or str(result)
            raise GraphAuthError(f"Device code authentication failed: {detail}")
        self._last_token_result = result
        return result

    def acquire_token_silent(self) -> dict[str, Any]:
        app = self._require_app()
        accounts = app.get_accounts()
        if not accounts:
            raise GraphNeedsReconnectError("No cached Microsoft account; reconnect required.")
        result = app.acquire_token_silent(self.scopes, account=accounts[0])
        _persist_token_cache(self._cache, self.cache_path)
        if not result or "access_token" not in result:
            raise GraphNeedsReconnectError("Cached Microsoft token expired; reconnect required.")
        self._last_token_result = result
        return result

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (_GraphTransientError, httpx.TransportError, httpx.TimeoutException)
        ),
    )
    def _graph_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = self.acquire_token_silent()
        url = f"{GRAPH_BASE_URL}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        with httpx.Client(timeout=20.0) as client:
            resp = client.request(method, url, headers=headers, json=payload)

        if resp.status_code == 401:
            raise GraphNeedsReconnectError("Microsoft token rejected by Graph; reconnect required.")
        if resp.status_code in (429, 500, 502, 503, 504):
            raise _GraphTransientError(f"Transient Graph error {resp.status_code}")
        if resp.status_code >= 400:
            raise GraphRequestError(f"Graph {method} {path} failed: {resp.status_code}")
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    def graph_get(self, path: str) -> dict[str, Any]:
        try:
            return self._graph_request("GET", path)
        except (httpx.TransportError, httpx.TimeoutException, _GraphTransientError) as exc:
            raise GraphRequestError(str(exc)) from exc

    def graph_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._graph_request("POST", path, payload=payload)
        except (httpx.TransportError, httpx.TimeoutException, _GraphTransientError) as exc:
            raise GraphRequestError(str(exc)) from exc

    @property
    def granted_scopes(self) -> list[str]:
        return _scopes_from_token_result(self._last_token_result)


def ms_connection_state(settings: AppSettings | None = None) -> dict[str, Any]:
    """Return UI/API-friendly Microsoft connection status."""
    cfg = settings or AppSettings()
    client = MicrosoftGraphClient(cfg)
    state: dict[str, Any] = {
        "configured": client.is_configured,
        "state": "Not configured",
        "status": "not_configured",
        "requested_scopes": cfg.ms_scopes_list,
        "granted_scopes": [],
        "tenant_id": cfg.ms_tenant_id,
        "account_display_name": None,
        "error": None,
    }
    if not client.is_configured:
        state["error"] = "Set MS_TENANT_ID and MS_CLIENT_ID to enable Microsoft Graph."
        return state

    cached_account = client.get_cached_account()
    if cached_account:
        state["account_display_name"] = cached_account.get("display_name")
        state["tenant_id"] = cached_account.get("tenant_id") or state["tenant_id"]

    try:
        me = client.graph_get("/me")
    except GraphNeedsReconnectError:
        state["state"] = "Expired"
        state["status"] = "expired"
        state["error"] = "Reconnect required."
    except GraphRequestError as exc:
        state["state"] = "Error"
        state["status"] = "error"
        state["error"] = str(exc)
    else:
        state["state"] = "Connected"
        state["status"] = "connected"
        state["account_display_name"] = (
            me.get("displayName")
            or me.get("userPrincipalName")
            or state["account_display_name"]
        )
        state["tenant_id"] = _token_tenant_id(client._last_token_result or {}) or state[
            "tenant_id"
        ]
        state["granted_scopes"] = client.granted_scopes or state["requested_scopes"]

    return state


def _complete_device_flow_in_background(
    session_id: str,
    *,
    settings: AppSettings,
    flow: dict[str, Any],
) -> None:
    status = "connected"
    error: str | None = None
    account_display_name: str | None = None
    tenant_id: str | None = None
    granted_scopes: list[str] = []
    try:
        client = MicrosoftGraphClient(settings=settings)
        result = client.acquire_token_by_device_flow(flow)
        account_display_name = _token_display_name(result)
        tenant_id = _token_tenant_id(result) or settings.ms_tenant_id
        granted_scopes = _scopes_from_token_result(result) or settings.ms_scopes_list
    except Exception as exc:  # pragma: no cover - tested via API surface
        status = "error"
        error = str(exc)

    with _DEVICE_FLOW_LOCK:
        now = time.time()
        _prune_sessions_locked(now)
        session = _DEVICE_FLOW_SESSIONS.get(session_id)
        if session is None:
            return
        session.status = status
        session.error = error
        session.account_display_name = account_display_name
        session.tenant_id = tenant_id
        session.granted_scopes = granted_scopes
        session.finished_at = now


def start_device_flow_session(
    settings: AppSettings | None = None,
    *,
    reconnect: bool = False,
) -> dict[str, Any]:
    cfg = settings or AppSettings()
    with _DEVICE_FLOW_LOCK:
        now = time.time()
        _prune_sessions_locked(now)
        existing = _first_pending_session_locked()
        if existing is not None:
            existing_id, existing_session = existing
            payload = _session_payload(existing_id, existing_session)
            payload["reused"] = True
            return payload
        if len(_pending_session_ids_locked()) >= _MAX_PENDING_DEVICE_FLOWS:
            raise GraphDeviceFlowLimitError(
                "Too many pending Microsoft connect sessions; try again shortly."
            )

    client = MicrosoftGraphClient(settings=cfg)
    if reconnect:
        client.clear_cache()
        client = MicrosoftGraphClient(settings=cfg)
    flow = client.initiate_device_flow()
    session_id = uuid4().hex
    session = _DeviceFlowSession(flow=flow)
    with _DEVICE_FLOW_LOCK:
        now = time.time()
        _prune_sessions_locked(now)
        existing = _first_pending_session_locked()
        if existing is not None:
            existing_id, existing_session = existing
            payload = _session_payload(existing_id, existing_session)
            payload["reused"] = True
            return payload
        if len(_pending_session_ids_locked()) >= _MAX_PENDING_DEVICE_FLOWS:
            raise GraphDeviceFlowLimitError(
                "Too many pending Microsoft connect sessions; try again shortly."
            )
        _DEVICE_FLOW_SESSIONS[session_id] = session

    thread = threading.Thread(
        target=_complete_device_flow_in_background,
        kwargs={"session_id": session_id, "settings": cfg, "flow": flow},
        daemon=True,
    )
    thread.start()
    payload = _session_payload(session_id, session)
    payload["reused"] = False
    return payload


def get_device_flow_session(session_id: str) -> dict[str, Any]:
    with _DEVICE_FLOW_LOCK:
        now = time.time()
        _prune_sessions_locked(now)
        session = _DEVICE_FLOW_SESSIONS.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return _session_payload(session_id, session)


__all__ = [
    "GraphAuthError",
    "GraphDeviceFlowLimitError",
    "GraphNeedsReconnectError",
    "GraphNotConfiguredError",
    "GraphRequestError",
    "MicrosoftGraphClient",
    "get_device_flow_session",
    "ms_connection_state",
    "start_device_flow_session",
]
