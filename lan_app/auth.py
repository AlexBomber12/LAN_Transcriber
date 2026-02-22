from __future__ import annotations

import secrets

from fastapi import Request, Response

from .config import AppSettings

AUTH_COOKIE_NAME = "lan_api_auth"
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = value.strip()
    return token or None


def expected_bearer_token(settings: AppSettings) -> str | None:
    return _normalize_token(settings.api_bearer_token)


def auth_enabled(settings: AppSettings) -> bool:
    return expected_bearer_token(settings) is not None


def _token_from_authorization_header(request: Request) -> str | None:
    raw = request.headers.get("Authorization")
    if not raw:
        return None
    scheme, sep, token = raw.partition(" ")
    if not sep or scheme.lower() != "bearer":
        return None
    return _normalize_token(token)


def _token_from_cookie(request: Request) -> str | None:
    return _normalize_token(request.cookies.get(AUTH_COOKIE_NAME))


def request_is_authenticated(request: Request, settings: AppSettings) -> bool:
    expected = expected_bearer_token(settings)
    if expected is None:
        return True

    candidates = (
        _token_from_authorization_header(request),
        _token_from_cookie(request),
    )
    for candidate in candidates:
        if candidate and secrets.compare_digest(candidate, expected):
            return True
    return False


def request_requires_auth(request: Request) -> bool:
    path = request.url.path
    method = request.method.upper()
    if method not in _MUTATING_METHODS:
        return False
    if path == "/ui/login":
        return False
    return True


def cookie_secure_flag(request: Request) -> bool:
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if forwarded_proto:
        scheme = forwarded_proto.split(",", 1)[0].strip().lower()
    else:
        scheme = str(request.url.scheme or "").strip().lower()
    return scheme == "https"


def set_auth_cookie(response: Response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/",
    )


def safe_next_path(next_path: str | None, *, default: str = "/ui") -> str:
    target = (next_path or "").strip()
    if not target or not target.startswith("/") or target.startswith("//"):
        return default
    return target


__all__ = [
    "AUTH_COOKIE_NAME",
    "auth_enabled",
    "clear_auth_cookie",
    "cookie_secure_flag",
    "expected_bearer_token",
    "request_is_authenticated",
    "request_requires_auth",
    "safe_next_path",
    "set_auth_cookie",
]
