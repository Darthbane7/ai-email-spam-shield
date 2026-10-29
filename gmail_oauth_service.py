"""
Google OAuth 2.0 + Gmail REST API v1 Multi-User Service (100% Free Stack)
Handles:
1. "Sign in with Google" OAuth 2.0 authorization & offline refresh_token renewal
2. Multi-user persistence (Supports Free Upstash Redis REST API in Vercel production
   with automatic SQLite fallback for local development)
3. Gmail REST API v1 scanning & custom 'AI-Spam' label creation/attachment (with
   badge colors that sync immediately to the Gmail Android App)
"""

import base64
from datetime import datetime
import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import joblib
import requests
from dotenv import load_dotenv

from preprocessing import preprocess_text

load_dotenv()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"

OAUTH_SCOPES = [
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.modify",
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(ROOT_DIR, "models", "spam_artifacts.joblib")

# Writable SQLite path (uses /tmp on Vercel serverless, or data/ locally)
IS_VERCEL = bool(os.environ.get("VERCEL"))
SQLITE_DB_PATH = (
    "/tmp/saas_users.db"
    if IS_VERCEL
    else os.path.join(ROOT_DIR, "data", "saas_users.db")
)

_CACHED_ARTIFACTS = None


def get_ml_artifacts():
    """Load and cache the trained TF-IDF vectorizer and ML classifiers in memory."""
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is None:
        _CACHED_ARTIFACTS = joblib.load(ARTIFACTS_PATH)
    return _CACHED_ARTIFACTS


# ============================================================================
# 1. MULTI-USER STORAGE (Free Upstash Redis REST API or Local SQLite Fallback)
# ============================================================================

def _init_sqlite() -> None:
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                name TEXT,
                picture TEXT,
                access_token TEXT,
                refresh_token TEXT,
                label_name TEXT DEFAULT 'AI-Spam',
                move_to_spam INTEGER DEFAULT 0,
                auto_filter_enabled INTEGER DEFAULT 1,
                total_scanned INTEGER DEFAULT 0,
                spam_detected INTEGER DEFAULT 0,
                last_scanned_at TEXT,
                recent_logs TEXT DEFAULT '[]',
                created_at TEXT
            )
            """
        )
        conn.commit()


def _use_upstash() -> bool:
    return bool(
        os.getenv("UPSTASH_REDIS_REST_URL") and os.getenv("UPSTASH_REDIS_REST_TOKEN")
    )


def _upstash_cmd(command: list):
    url = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=command,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("result")


def save_user(user_data: dict) -> dict:
    """Create or update a registered user and preserve their refresh_token if not re-issued."""
    email_addr = user_data["email"].lower().strip()
    existing = get_user(email_addr) or {}

    merged = {
        "email": email_addr,
        "name": user_data.get("name") or existing.get("name") or email_addr.split("@")[0],
        "picture": user_data.get("picture") or existing.get("picture") or "",
        "access_token": user_data.get("access_token") or existing.get("access_token") or "",
        "refresh_token": user_data.get("refresh_token") or existing.get("refresh_token") or "",
        "label_name": user_data.get("label_name") or existing.get("label_name") or "AI-Spam",
        "move_to_spam": bool(user_data.get("move_to_spam", existing.get("move_to_spam", False))),
        "auto_filter_enabled": bool(
            user_data.get("auto_filter_enabled", existing.get("auto_filter_enabled", True))
        ),
        "total_scanned": int(user_data.get("total_scanned", existing.get("total_scanned", 0))),
        "spam_detected": int(user_data.get("spam_detected", existing.get("spam_detected", 0))),
        "last_scanned_at": user_data.get("last_scanned_at") or existing.get("last_scanned_at") or "Never",
        "recent_logs": user_data.get("recent_logs", existing.get("recent_logs", [])),
        "created_at": existing.get("created_at") or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }

    if _use_upstash():
        _upstash_cmd(["HSET", "ai_spam_users", email_addr, json.dumps(merged)])
        return merged

    _init_sqlite()
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO users (
                email, name, picture, access_token, refresh_token,
                label_name, move_to_spam, auto_filter_enabled,
                total_scanned, spam_detected, last_scanned_at, recent_logs, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                name=excluded.name,
                picture=excluded.picture,
                access_token=excluded.access_token,
                refresh_token=excluded.refresh_token,
                label_name=excluded.label_name,
                move_to_spam=excluded.move_to_spam,
                auto_filter_enabled=excluded.auto_filter_enabled,
                total_scanned=excluded.total_scanned,
                spam_detected=excluded.spam_detected,
                last_scanned_at=excluded.last_scanned_at,
                recent_logs=excluded.recent_logs
            """,
            (
                merged["email"],
                merged["name"],
                merged["picture"],
                merged["access_token"],
                merged["refresh_token"],
                merged["label_name"],
                int(merged["move_to_spam"]),
                int(merged["auto_filter_enabled"]),
                merged["total_scanned"],
                merged["spam_detected"],
                merged["last_scanned_at"],
                json.dumps(merged["recent_logs"]),
                merged["created_at"],
            ),
        )
        conn.commit()
    return merged


def get_user(email_addr: str) -> Optional[dict]:
    """Retrieve a single registered user by email address."""
    email_addr = email_addr.lower().strip()
    if _use_upstash():
        raw = _upstash_cmd(["HGET", "ai_spam_users", email_addr])
        return json.loads(raw) if raw else None

    _init_sqlite()
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email_addr,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["move_to_spam"] = bool(d["move_to_spam"])
        d["auto_filter_enabled"] = bool(d["auto_filter_enabled"])
        try:
            d["recent_logs"] = json.loads(d.get("recent_logs") or "[]")
        except Exception:
            d["recent_logs"] = []
        return d


def list_active_users() -> List[dict]:
    """Return all registered users with auto_filter_enabled=True for background cron scanning."""
    if _use_upstash():
        raw_map = _upstash_cmd(["HGETALL", "ai_spam_users"]) or []
        users = []
        # HGETALL returns [key1, val1, key2, val2, ...]
        for i in range(1, len(raw_map), 2):
            try:
                u = json.loads(raw_map[i])
                if u.get("auto_filter_enabled", True):
                    users.append(u)
            except Exception:
                pass
        return users

    _init_sqlite()
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM users WHERE auto_filter_enabled = 1").fetchall()
        users = []
        for r in rows:
            d = dict(r)
            d["move_to_spam"] = bool(d["move_to_spam"])
            d["auto_filter_enabled"] = bool(d["auto_filter_enabled"])
            try:
                d["recent_logs"] = json.loads(d.get("recent_logs") or "[]")
            except Exception:
                d["recent_logs"] = []
            users.append(d)
        return users


# ============================================================================
# 2. GOOGLE OAUTH 2.0 FLOW
# ============================================================================

def build_google_auth_url(redirect_uri: str, state: str = "login") -> Tuple[Optional[str], Optional[str]]:
    """Build the Google OAuth 2.0 consent screen URL."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        return None, "GOOGLE_CLIENT_ID is not configured in environment variables."

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(OAUTH_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}", None


def exchange_code_for_tokens(code: str, redirect_uri: str) -> dict:
    """Exchange authorization code for access_token, refresh_token, and Google profile."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

    token_resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    token_resp.raise_for_status()
    tokens = token_resp.json()

    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token", "")

    profile_resp = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    profile_resp.raise_for_status()
    profile = profile_resp.json()

    return {
        "email": profile["email"],
        "name": profile.get("name", ""),
        "picture": profile.get("picture", ""),
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


def refresh_google_access_token(refresh_token: str) -> Optional[str]:
    """Use a user's stored refresh_token to obtain a fresh access_token."""
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret or not refresh_token:
        return None

    resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    if resp.status_code != 200:
        return None
    return resp.json().get("access_token")


# ============================================================================
# 3. GMAIL REST API v1 SCANNING & LABELING (Syncs to Android Gmail App)
# ============================================================================

def _decode_base64url(data_str: str) -> str:
    """Decode base64url-encoded Gmail MIME payload into UTF-8 text."""
    if not data_str:
        return ""
    padding = "=" * ((4 - len(data_str) % 4) % 4)
    try:
        raw_bytes = base64.urlsafe_b64decode(data_str + padding)
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body_from_gmail_payload(payload: dict) -> str:
    """Recursively extract plain/HTML body text from a Gmail API message payload."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime_type == "text/plain" and body_data:
        return _decode_base64url(body_data)

    if mime_type == "text/html" and body_data:
        raw_html = _decode_base64url(body_data)
        raw_html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        return re.sub(r"<[^>]+>", " ", raw_html)

    plain_parts = []
    html_parts = []
    for part in payload.get("parts", []) or []:
        part_mime = part.get("mimeType", "")
        if part_mime == "text/plain":
            plain_parts.append(_extract_body_from_gmail_payload(part))
        elif part_mime == "text/html":
            html_parts.append(_extract_body_from_gmail_payload(part))
        elif part_mime.startswith("multipart/"):
            plain_parts.append(_extract_body_from_gmail_payload(part))

    combined_plain = "\n".join(p for p in plain_parts if p).strip()
    if combined_plain:
        return combined_plain
    return "\n".join(h for h in html_parts if h).strip()


def ensure_gmail_api_label(headers: dict, label_name: str) -> Optional[str]:
    """
    Find or create the custom Gmail label (e.g. 'AI-Spam') via Gmail REST API
    and return its Gmail label ID. Sets a high-contrast red badge for Android/Web.
    """
    resp = requests.get(f"{GMAIL_API_BASE}/labels", headers=headers, timeout=10)
    if resp.status_code != 200:
        return None

    labels = resp.json().get("labels", [])
    for lbl in labels:
        if lbl.get("name", "").lower() == label_name.lower():
            return lbl.get("id")

    create_payload = {
        "name": label_name,
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
        "color": {
            "backgroundColor": "#cc3a21",
            "textColor": "#ffffff",
        },
    }
    create_resp = requests.post(
        f"{GMAIL_API_BASE}/labels",
        headers=headers,
        json=create_payload,
        timeout=10,
    )
    if create_resp.status_code in (200, 201):
        return create_resp.json().get("id")
    return None


def scan_user_gmail_oauth(
    user: dict,
    max_emails: int = 15,
    model_name: Optional[str] = None,
) -> Tuple[List[dict], Optional[str]]:
    """
    Scan a registered user's Gmail inbox using their OAuth2 token, classify
    each message with the trained ML model, and apply the 'AI-Spam' label.
    """
    access_token = user.get("access_token", "")
    refresh_token = user.get("refresh_token", "")
    label_name = user.get("label_name", "AI-Spam")
    move_to_spam = bool(user.get("move_to_spam", False))

    headers = {"Authorization": f"Bearer {access_token}"}

    # Test token validity; refresh automatically if expired
    profile_check = requests.get(f"{GMAIL_API_BASE}/profile", headers=headers, timeout=10)
    if profile_check.status_code == 401 and refresh_token:
        new_token = refresh_google_access_token(refresh_token)
        if not new_token:
            return [], "Google OAuth session expired. Please sign in with Google again."
        access_token = new_token
        user["access_token"] = new_token
        save_user(user)
        headers = {"Authorization": f"Bearer {access_token}"}
    elif profile_check.status_code != 200:
        return [], f"Gmail API error ({profile_check.status_code}): {profile_check.text}"

    label_id = ensure_gmail_api_label(headers, label_name)

    # Query recent Inbox messages that haven't already been tagged with our label
    query = f'in:inbox -label:"{label_name}"'
    list_resp = requests.get(
        f"{GMAIL_API_BASE}/messages",
        headers=headers,
        params={"q": query, "maxResults": max_emails},
        timeout=12,
    )
    if list_resp.status_code != 200:
        return [], f"Failed to list messages: {list_resp.text}"

    messages_meta = list_resp.json().get("messages", [])
    if not messages_meta:
        user["last_scanned_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        save_user(user)
        return [], None

    artifacts = get_ml_artifacts()
    vectorizer = artifacts["vectorizer"]
    chosen_model_name = model_name or artifacts["best_model_name"]
    model = artifacts["models"][chosen_model_name]

    scanned_results: List[dict] = []
    new_spam_count = 0

    for m_meta in messages_meta:
        msg_id = m_meta["id"]
        msg_resp = requests.get(
            f"{GMAIL_API_BASE}/messages/{msg_id}",
            headers=headers,
            params={"format": "full"},
            timeout=10,
        )
        if msg_resp.status_code != 200:
            continue

        msg_json = msg_resp.json()
        payload = msg_json.get("payload", {})
        hdr_list = payload.get("headers", [])
        hdr_map = {h["name"].lower(): h["value"] for h in hdr_list if "name" in h and "value" in h}

        subject = hdr_map.get("subject", "(No Subject)")
        sender = hdr_map.get("from", "(Unknown Sender)")
        date_str = hdr_map.get("date", "")
        snippet = msg_json.get("snippet", "")

        body_text = _extract_body_from_gmail_payload(payload) or snippet
        combined = f"{subject}\n{body_text}".strip()

        clean_text = preprocess_text(combined)
        X_vec = vectorizer.transform([clean_text])
        pred = int(model.predict(X_vec)[0])
        spam_prob = float(model.predict_proba(X_vec)[0][1])

        action_taken = "Kept in Inbox (Ham)"
        if pred == 1:
            new_spam_count += 1
            modify_body: Dict[str, List[str]] = {"addLabelIds": [], "removeLabelIds": []}
            if label_id:
                modify_body["addLabelIds"].append(label_id)
            if move_to_spam:
                modify_body["addLabelIds"].append("SPAM")
                modify_body["removeLabelIds"].append("INBOX")

            if modify_body["addLabelIds"] or modify_body["removeLabelIds"]:
                requests.post(
                    f"{GMAIL_API_BASE}/messages/{msg_id}/modify",
                    headers=headers,
                    json=modify_body,
                    timeout=10,
                )
            action_taken = (
                f"Labeled '{label_name}' & Moved to Spam"
                if move_to_spam
                else f"Labeled '{label_name}' (Kept in Inbox)"
            )

        scanned_results.append(
            {
                "id": msg_id,
                "from": sender,
                "subject": subject,
                "date": date_str,
                "prediction": "SPAM" if pred == 1 else "HAM",
                "spam_probability": round(spam_prob, 4),
                "action_taken": action_taken,
                "scanned_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            }
        )

    user["total_scanned"] = int(user.get("total_scanned", 0)) + len(scanned_results)
    user["spam_detected"] = int(user.get("spam_detected", 0)) + new_spam_count
    user["last_scanned_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    existing_logs = user.get("recent_logs", [])
    user["recent_logs"] = (scanned_results + existing_logs)[:40]
    save_user(user)

    return scanned_results, None
