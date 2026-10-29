"""
Vercel Serverless FastAPI Backend + Multi-User Google OAuth2 SaaS UI
Routes:
- GET  /                         -> SaaS Web Dashboard (Sign in with Google, Live Classifier, Inbox Scanner)
- POST /api/predict              -> Classify raw email text using the trained ML models
- GET  /api/auth/google/login    -> Redirect user to Google OAuth 2.0 Consent Screen
- GET  /api/auth/google/callback -> Exchange OAuth code, register user, and start session
- GET  /api/me                   -> Return logged-in user's profile, settings, and scan history
- POST /api/settings             -> Update user's custom Gmail label ('AI-Spam') & auto-filter settings
- POST /api/gmail/scan           -> Scan logged-in user's Gmail inbox immediately
- GET  /api/cron/scan-all        -> Background Cron endpoint to filter emails for all registered users
"""

import hashlib
import hmac
import os
import sys
from typing import Optional
from urllib.parse import parse_qsl, urlencode
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

# Ensure project root is on sys.path when invoked inside Vercel's /api directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gmail_oauth_service import (
    build_google_auth_url,
    exchange_code_for_tokens,
    get_ml_artifacts,
    get_user,
    list_active_users,
    save_user,
    scan_user_gmail_oauth,
)
from preprocessing import preprocess_steps_breakdown

app = FastAPI(
    title="AI Email Spam Filter SaaS (Google OAuth2 + Android Sync)",
    version="2.0.1",
)


class VercelPathRewriteMiddleware:
    """
    Normalizes incoming paths when Vercel rewrites `/(.*)` to `/api/index.py?__path=/$1`
    or `/api/index.py`, ensuring FastAPI routes match on both Vercel and localhost.
    """

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            raw_qs = scope.get("query_string", b"").decode("utf-8", errors="ignore")
            if "__path=" in raw_qs:
                pairs = parse_qsl(raw_qs, keep_blank_values=True)
                rewritten_path = None
                remaining_pairs = []
                for k, v in pairs:
                    if k == "__path" and rewritten_path is None:
                        rewritten_path = v
                    else:
                        remaining_pairs.append((k, v))
                if rewritten_path:
                    if not rewritten_path.startswith("/"):
                        rewritten_path = "/" + rewritten_path
                    scope = dict(scope)
                    scope["path"] = rewritten_path
                    scope["raw_path"] = rewritten_path.encode("utf-8")
                    scope["query_string"] = urlencode(remaining_pairs).encode("utf-8")
            elif scope.get("path") in ("/api/index.py", "/api/index"):
                scope = dict(scope)
                scope["path"] = "/"
                scope["raw_path"] = b"/"

        await self.asgi_app(scope, receive, send)


app.add_middleware(VercelPathRewriteMiddleware)

SESSION_SECRET = os.getenv("SESSION_SECRET", "ai-spam-filter-free-saas-secret-key-2026")


# ============================================================================
# SESSION SIGNING HELPERS (Stateless HMAC Cookie for Serverless)
# ============================================================================

def sign_session_email(email_addr: str) -> str:
    email_clean = email_addr.lower().strip()
    sig = hmac.new(
        SESSION_SECRET.encode("utf-8"),
        email_clean.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{email_clean}|{sig}"


def verify_session_cookie(cookie_val: Optional[str]) -> Optional[str]:
    if not cookie_val or "|" not in cookie_val:
        return None
    email_clean, sig = cookie_val.rsplit("|", 1)
    expected = hmac.new(
        SESSION_SECRET.encode("utf-8"),
        email_clean.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if hmac.compare_digest(sig, expected):
        return email_clean
    return None


def get_base_url(request: Request) -> str:
    """Resolve public base URL on Vercel or localhost."""
    env_url = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if env_url:
        return env_url
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost:8000"))
    return f"{proto}://{host}"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class PredictRequest(BaseModel):
    text: str
    model_name: Optional[str] = None


class UserSettingsRequest(BaseModel):
    label_name: str = "AI-Spam"
    move_to_spam: bool = False
    auto_filter_enabled: bool = True


class ScanInboxRequest(BaseModel):
    max_emails: int = 15


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    """Run NLP preprocessing, TF-IDF vectorization, and multi-model spam classification."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    artifacts = get_ml_artifacts()
    vectorizer = artifacts["vectorizer"]
    models = artifacts["models"]
    best_model_name = artifacts["best_model_name"]
    chosen_name = req.model_name if req.model_name in models else best_model_name

    steps = preprocess_steps_breakdown(req.text)
    clean_text = steps["5_stemmed_final"]
    X_vec = vectorizer.transform([clean_text])

    primary_model = models[chosen_name]
    spam_prob = float(primary_model.predict_proba(X_vec)[0][1])
    pred = 1 if spam_prob > 0.86 else 0

    consensus = []
    for m_name, m_obj in models.items():
        prob_i = float(m_obj.predict_proba(X_vec)[0][1])
        p_i = 1 if prob_i > 0.86 else 0
        consensus.append(
            {
                "model": m_name,
                "prediction": "SPAM" if p_i == 1 else "HAM",
                "spam_probability": round(prob_i, 4),
            }
        )

    # Top TF-IDF tokens
    nonzero_idx = X_vec.nonzero()[1]
    top_tokens = []
    if len(nonzero_idx) > 0:
        feature_names = np.array(vectorizer.get_feature_names_out())
        scores = X_vec.data
        order = np.argsort(scores)[::-1][:8]
        for idx in order:
            top_tokens.append(
                {
                    "token": str(feature_names[nonzero_idx[idx]]),
                    "score": round(float(scores[idx]), 4),
                }
            )

    return {
        "prediction": "SPAM" if pred == 1 else "HAM",
        "spam_probability": round(spam_prob, 4),
        "model_used": chosen_name,
        "consensus": consensus,
        "top_tokens": top_tokens,
        "clean_text": clean_text,
    }


@app.get("/api/auth/google/login")
async def google_login(request: Request):
    """Redirect the user to Google's OAuth 2.0 consent screen."""
    redirect_uri = f"{get_base_url(request)}/api/auth/google/callback"
    auth_url, err = build_google_auth_url(redirect_uri=redirect_uri)
    if err:
        return RedirectResponse(url=f"/?error={err}")
    return RedirectResponse(url=auth_url)


@app.get("/api/auth/google/callback")
async def google_callback(request: Request, code: Optional[str] = None, error: Optional[str] = None):
    """Handle Google OAuth 2.0 callback, save user + refresh_token, and set session cookie."""
    if error:
        return RedirectResponse(url=f"/?error=Google+OAuth+denied:+{error}")
    if not code:
        return RedirectResponse(url="/?error=Missing+authorization+code")

    redirect_uri = f"{get_base_url(request)}/api/auth/google/callback"
    try:
        oauth_user = exchange_code_for_tokens(code=code, redirect_uri=redirect_uri)
        saved = save_user(oauth_user)
        resp = RedirectResponse(url="/?connected=1")
        resp.set_cookie(
            key="ai_spam_session",
            value=sign_session_email(saved["email"]),
            httponly=True,
            max_age=60 * 60 * 24 * 30,
            samesite="lax",
        )
        return resp
    except Exception as exc:
        return RedirectResponse(url=f"/?error=OAuth+exchange+failed:+{str(exc)[:120]}")


@app.post("/api/auth/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("ai_spam_session")
    return resp


@app.get("/api/me")
async def api_me(request: Request):
    """Return current user's session status, profile, and model metrics."""
    artifacts = get_ml_artifacts()
    oauth_configured = bool(
        os.getenv("GOOGLE_CLIENT_ID", "").strip()
        and os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    )
    redirect_uri = f"{get_base_url(request)}/api/auth/google/callback"

    email_addr = verify_session_cookie(request.cookies.get("ai_spam_session"))
    user = get_user(email_addr) if email_addr else None

    safe_user = None
    if user:
        safe_user = {
            "email": user["email"],
            "name": user["name"],
            "picture": user["picture"],
            "label_name": user["label_name"],
            "move_to_spam": user["move_to_spam"],
            "auto_filter_enabled": user["auto_filter_enabled"],
            "total_scanned": user["total_scanned"],
            "spam_detected": user["spam_detected"],
            "last_scanned_at": user["last_scanned_at"],
            "recent_logs": user.get("recent_logs", [])[:20],
        }

    return {
        "authenticated": bool(safe_user),
        "user": safe_user,
        "oauth_configured": oauth_configured,
        "oauth_redirect_uri": redirect_uri,
        "best_model_name": artifacts["best_model_name"],
        "metrics": artifacts["metrics"],
    }


@app.post("/api/settings")
async def api_update_settings(request: Request, body: UserSettingsRequest):
    email_addr = verify_session_cookie(request.cookies.get("ai_spam_session"))
    if not email_addr:
        raise HTTPException(status_code=401, detail="Not signed in.")
    user = get_user(email_addr)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user["label_name"] = body.label_name.strip() or "AI-Spam"
    user["move_to_spam"] = body.move_to_spam
    user["auto_filter_enabled"] = body.auto_filter_enabled
    updated = save_user(user)
    return {
        "ok": True,
        "label_name": updated["label_name"],
        "move_to_spam": updated["move_to_spam"],
        "auto_filter_enabled": updated["auto_filter_enabled"],
    }


@app.post("/api/gmail/scan")
async def api_scan_my_inbox(request: Request, body: ScanInboxRequest):
    """Trigger an immediate Gmail API scan for the currently logged-in user."""
    email_addr = verify_session_cookie(request.cookies.get("ai_spam_session"))
    if not email_addr:
        raise HTTPException(status_code=401, detail="Please sign in with Google first.")
    user = get_user(email_addr)
    if not user:
        raise HTTPException(status_code=404, detail="User record not found.")

    results, err = scan_user_gmail_oauth(user, max_emails=min(max(body.max_emails, 1), 50))
    if err:
        raise HTTPException(status_code=400, detail=err)

    updated_user = get_user(email_addr)
    return {
        "ok": True,
        "scanned_count": len(results),
        "spam_flagged": sum(1 for r in results if r["prediction"] == "SPAM"),
        "results": results,
        "user_stats": {
            "total_scanned": updated_user["total_scanned"],
            "spam_detected": updated_user["spam_detected"],
            "last_scanned_at": updated_user["last_scanned_at"],
        },
    }


@app.api_route("/api/cron/scan-all", methods=["GET", "POST"])
async def api_cron_scan_all(request: Request):
    """
    Automated background endpoint invoked by Vercel Cron or cron-job.org (Free).
    Loops through all registered users with auto_filter_enabled=True, scans their
    new Gmail messages, and attaches the 'AI-Spam' label so it syncs to Android!
    """
    cron_secret = os.getenv("CRON_SECRET", "").strip()
    if cron_secret:
        auth_hdr = request.headers.get("authorization", "")
        query_key = request.query_params.get("key", "")
        if auth_hdr != f"Bearer {cron_secret}" and query_key != cron_secret:
            raise HTTPException(status_code=401, detail="Unauthorized cron trigger.")

    active_users = list_active_users()
    summary = []
    for u in active_users:
        results, err = scan_user_gmail_oauth(u, max_emails=15)
        summary.append(
            {
                "email": u["email"],
                "scanned": len(results),
                "spam_flagged": sum(1 for r in results if r["prediction"] == "SPAM"),
                "error": err,
            }
        )

    return {
        "ok": True,
        "users_processed": len(active_users),
        "summary": summary,
    }


# ============================================================================
# SAAS WEB FRONTEND (Single-Page Dashboard at GET /)
# ============================================================================

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Email Spam Shield — Free Gmail & Android Auto-Filter</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans">
  <!-- Top Navbar -->
  <header class="border-b border-slate-800 bg-slate-900/70 backdrop-blur sticky top-0 z-30">
    <div class="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-2xl">🛡️</span>
        <div>
          <h1 class="font-bold text-lg tracking-tight text-white">AI Spam Shield</h1>
          <p class="text-xs text-slate-400">Free Gmail & Android App Auto-Filter Service</p>
        </div>
      </div>
      <div id="nav-auth-area" class="flex items-center gap-3">
        <a href="/api/auth/google/login"
           class="inline-flex items-center gap-2 bg-white text-slate-900 hover:bg-slate-200 font-semibold text-sm px-4 py-2 rounded-lg shadow transition">
          <svg class="w-4 h-4" viewBox="0 0 24 24">
            <path fill="#EA4335" d="M12 10.2v3.9h5.5c-.2 1.3-1.6 3.8-5.5 3.8-3.3 0-6-2.7-6-6s2.7-6 6-6c1.9 0 3.1.8 3.8 1.5l2.6-2.5C16.7 3.4 14.6 2.5 12 2.5 6.8 2.5 2.5 6.8 2.5 12s4.3 9.5 9.5 9.5c5.5 0 9.1-3.9 9.1-9.3 0-.6-.1-1.1-.2-1.6H12z"/>
          </svg>
          Sign in with Google
        </a>
      </div>
    </div>
  </header>

  <main class="max-w-6xl mx-auto px-4 py-8 space-y-8">
    <!-- Alert Banner -->
    <div id="alert-box" class="hidden rounded-xl border p-4 text-sm"></div>

    <!-- Hero / Multi-User SaaS Onboarding -->
    <section id="hero-section" class="rounded-2xl bg-gradient-to-br from-indigo-950/80 via-slate-900 to-slate-900 border border-indigo-500/30 p-6 md:p-8 shadow-xl">
      <div class="grid md:grid-cols-3 gap-6 items-center">
        <div class="md:col-span-2 space-y-3">
          <span class="inline-block text-xs font-semibold uppercase tracking-wider px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            100% Free Cloud Service • Works on Android & Web
          </span>
          <h2 class="text-2xl md:text-3xl font-extrabold text-white">
            Automatically label spam in your Gmail & Android App with Machine Learning
          </h2>
          <p class="text-slate-300 text-sm leading-relaxed">
            Connect your Gmail with one click using <strong>Google OAuth 2.0</strong>. Our TF-IDF + Support Vector Machine (SVM) pipeline scans incoming emails in the background and attaches a bright <code class="px-1.5 py-0.5 rounded bg-red-500/20 text-red-300 font-mono">AI-Spam</code> badge while keeping messages safely in your Inbox—syncing automatically to your <strong>Android Gmail app</strong>.
          </p>
          <div class="pt-2 flex flex-wrap gap-3">
            <a href="/api/auth/google/login"
               class="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-5 py-2.5 rounded-xl shadow-lg transition">
              🚀 Connect Gmail with Google OAuth2
            </a>
            <button onclick="document.getElementById('setup-guide').classList.toggle('hidden')"
               class="inline-flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium px-4 py-2.5 rounded-xl border border-slate-700 transition text-sm">
              ⚙️ Free Vercel + Google OAuth Setup Guide
            </button>
          </div>
        </div>
        <div class="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-2.5 text-xs">
          <div class="font-semibold text-slate-200 flex items-center justify-between">
            <span>📱 Android Gmail App Preview</span>
            <span class="text-emerald-400">● Live Sync</span>
          </div>
          <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex items-start justify-between gap-2">
            <div>
              <div class="font-semibold text-white">Prize Claim Center</div>
              <div class="text-slate-400 truncate max-w-[180px]">URGENT! You won a $1000 gift card...</div>
            </div>
            <span class="px-2 py-0.5 rounded bg-red-600 text-white font-bold text-[10px] shrink-0">AI-Spam</span>
          </div>
          <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex items-start justify-between gap-2">
            <div>
              <div class="font-semibold text-white">Alex Rivera</div>
              <div class="text-slate-400 truncate max-w-[180px]">Are we still on for coffee Friday?</div>
            </div>
            <span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold text-[10px] shrink-0">Inbox</span>
          </div>
        </div>
      </div>

      <!-- Collapsible Admin Setup Guide -->
      <div id="setup-guide" class="hidden mt-6 pt-6 border-t border-slate-800 text-sm space-y-3 text-slate-300">
        <h3 class="font-bold text-white text-base">🔑 How to Configure Free Google OAuth2 on Vercel (One-Time Admin Setup)</h3>
        <ol class="list-decimal list-inside space-y-2 text-xs md:text-sm">
          <li>Go to the free <a href="https://console.cloud.google.com/apis/credentials" target="_blank" class="text-indigo-400 underline">Google Cloud Console</a>, create a project, and enable the <strong>Gmail API</strong>.</li>
          <li>Under <strong>OAuth consent screen</strong>, add scope <code class="text-indigo-300">https://www.googleapis.com/auth/gmail.modify</code> and add your testers (or publish the app).</li>
          <li>Under <strong>Credentials ➔ Create Credentials ➔ OAuth Client ID (Web application)</strong>, add this exact Authorized Redirect URI:
            <code id="oauth-redirect-code" class="block mt-1 p-2 rounded bg-slate-950 border border-slate-800 text-emerald-400 font-mono select-all">http://localhost:8000/api/auth/google/callback</code>
          </li>
          <li>Add <code class="text-amber-300">GOOGLE_CLIENT_ID</code> and <code class="text-amber-300">GOOGLE_CLIENT_SECRET</code> to your Vercel Environment Variables (or local <code class="text-amber-300">.env</code>).</li>
          <li><strong>For 100% Free Multi-User Cloud Database on Vercel:</strong> Create a free Redis database at <a href="https://upstash.com" target="_blank" class="text-indigo-400 underline">Upstash.com</a> (no credit card required) and add <code class="text-amber-300">UPSTASH_REDIS_REST_URL</code> and <code class="text-amber-300">UPSTASH_REDIS_REST_TOKEN</code> to Vercel.</li>
          <li><strong>For 100% Free Every-5-Minute Background Scanning:</strong> Add <code class="text-emerald-400">https://your-app.vercel.app/api/cron/scan-all</code> to <a href="https://cron-job.org" target="_blank" class="text-indigo-400 underline">cron-job.org</a> (free).</li>
        </ol>
      </div>
    </section>

    <!-- Logged-In User Auto-Filter Dashboard (Hidden until signed in) -->
    <section id="user-dashboard" class="hidden space-y-6">
      <div class="grid sm:grid-cols-3 gap-4">
        <div class="rounded-xl bg-slate-900 border border-slate-800 p-4">
          <div class="text-xs text-slate-400">Total Emails Scanned</div>
          <div id="stat-scanned" class="text-2xl font-bold text-white mt-1">0</div>
        </div>
        <div class="rounded-xl bg-slate-900 border border-slate-800 p-4">
          <div class="text-xs text-slate-400">Spam Flagged with Label</div>
          <div id="stat-spam" class="text-2xl font-bold text-red-400 mt-1">0</div>
        </div>
        <div class="rounded-xl bg-slate-900 border border-slate-800 p-4">
          <div class="text-xs text-slate-400">Last Inbox Sync</div>
          <div id="stat-last-scan" class="text-sm font-semibold text-emerald-400 mt-2">Never</div>
        </div>
      </div>

      <div class="grid md:grid-cols-3 gap-6">
        <!-- Auto-Filter Settings Card -->
        <div class="rounded-2xl bg-slate-900 border border-slate-800 p-5 space-y-4">
          <h3 class="font-bold text-white flex items-center gap-2">⚙️ Gmail & Android Filter Settings</h3>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Custom Gmail Label Name</label>
            <input id="setting-label" type="text" value="AI-Spam"
                   class="w-full rounded-lg bg-slate-950 border border-slate-800 px-3 py-2 text-sm text-white" />
            <p class="text-[11px] text-slate-500 mt-1">Appears as a red badge in Gmail Web & Android app.</p>
          </div>
          <label class="flex items-center gap-2.5 text-sm cursor-pointer">
            <input id="setting-auto" type="checkbox" checked class="rounded text-indigo-600" />
            <span>Enable automatic background filtering</span>
          </label>
          <label class="flex items-center gap-2.5 text-sm cursor-pointer">
            <input id="setting-move" type="checkbox" class="rounded text-indigo-600" />
            <span>Move directly to Spam folder (hide from Inbox)</span>
          </label>
          <button onclick="saveUserSettings()"
                  class="w-full bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white font-semibold py-2 rounded-lg text-sm transition">
            💾 Save Filter Preferences
          </button>
        </div>

        <!-- Live Inbox Scanner Card -->
        <div class="md:col-span-2 rounded-2xl bg-slate-900 border border-slate-800 p-5 space-y-4">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 class="font-bold text-white">📬 Live Gmail Inbox Scanner</h3>
              <p class="text-xs text-slate-400">Scan your latest Inbox emails and apply the <code class="text-red-300">AI-Spam</code> label immediately.</p>
            </div>
            <button id="btn-scan-now" onclick="scanMyInbox()"
                    class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-4 py-2 rounded-xl text-sm shadow transition">
              🔄 Scan My Gmail Inbox Now
            </button>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs">
              <thead class="text-slate-400 border-b border-slate-800">
                <tr>
                  <th class="py-2 pr-3">Status</th>
                  <th class="py-2 pr-3">Confidence</th>
                  <th class="py-2 pr-3">From</th>
                  <th class="py-2 pr-3">Subject</th>
                  <th class="py-2">Gmail / Android Action</th>
                </tr>
              </thead>
              <tbody id="scan-logs-body" class="divide-y divide-slate-800/70">
                <tr>
                  <td colspan="5" class="py-6 text-center text-slate-500">
                    Click "Scan My Gmail Inbox Now" to classify your recent emails.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- Interactive Sandbox Classifier (Available to everyone) -->
    <section class="rounded-2xl bg-slate-900 border border-slate-800 p-6 space-y-4">
      <div class="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 class="font-bold text-lg text-white">🔍 Interactive Email Spam Classifier Sandbox</h3>
          <p class="text-xs text-slate-400">Test any email body right now using our trained TF-IDF + SVM, Logistic Regression & Naive Bayes models.</p>
        </div>
        <div class="flex gap-2 text-xs">
          <button onclick="loadSample('spam')" class="px-3 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-300 border border-red-500/20">
            Load Spam Sample
          </button>
          <button onclick="loadSample('ham')" class="px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/20">
            Load Ham Sample
          </button>
        </div>
      </div>

      <textarea id="sandbox-input" rows="4"
                placeholder="Paste an email subject and body here to test..."
                class="w-full rounded-xl bg-slate-950 border border-slate-800 p-3.5 text-sm text-white focus:outline-none focus:border-indigo-500"></textarea>

      <div class="flex items-center justify-between">
        <button onclick="runSandboxPrediction()"
                class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-5 py-2.5 rounded-xl text-sm transition">
          🚀 Classify Email Text
        </button>
        <span id="sandbox-model-badge" class="text-xs text-slate-400"></span>
      </div>

      <div id="sandbox-result" class="hidden pt-4 border-t border-slate-800 grid md:grid-cols-2 gap-4">
        <div id="sandbox-verdict" class="p-4 rounded-xl border"></div>
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="text-xs font-semibold text-slate-400">Multi-Model Consensus & Top Trigger Tokens</div>
          <div id="sandbox-consensus" class="space-y-1 text-xs"></div>
          <div id="sandbox-tokens" class="flex flex-wrap gap-1.5 pt-2"></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const SAMPLES = {
      spam: "URGENT! Congratulations! Your email address has been selected as the winner of a $1,000 Amazon Gift Card and a FREE luxury holiday! Click http://claim-prize-now.biz within 24 hours to claim your reward!",
      ham: "Hi team, please find attached the Q3 product roadmap and meeting notes from yesterday's sync. Let's meet tomorrow at 2:30 PM to finalize the deployment timeline."
    };

    function showAlert(msg, isError = false) {
      const box = document.getElementById('alert-box');
      box.className = `rounded-xl border p-4 text-sm ${
        isError
          ? 'bg-red-950/60 border-red-500/40 text-red-200'
          : 'bg-emerald-950/60 border-emerald-500/40 text-emerald-200'
      }`;
      box.textContent = msg;
      box.classList.remove('hidden');
    }

    function loadSample(type) {
      document.getElementById('sandbox-input').value = SAMPLES[type];
      runSandboxPrediction();
    }

    async function runSandboxPrediction() {
      const text = document.getElementById('sandbox-input').value.trim();
      if (!text) return;
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
      });
      if (!res.ok) return;
      const data = await res.json();

      document.getElementById('sandbox-result').classList.remove('hidden');
      const verdict = document.getElementById('sandbox-verdict');
      const pct = (data.spam_probability * 100).toFixed(1) + '%';
      if (data.prediction === 'SPAM') {
        verdict.className = 'p-4 rounded-xl border bg-red-950/40 border-red-500/40';
        verdict.innerHTML = `<div class="text-red-400 font-bold text-lg">🚨 SPAM DETECTED (${pct} Risk)</div>
          <p class="text-xs text-slate-300 mt-1">Flagged by <strong>${data.model_used}</strong>. In Gmail, this receives the <code class="text-red-300">AI-Spam</code> label.</p>`;
      } else {
        verdict.className = 'p-4 rounded-xl border bg-emerald-950/40 border-emerald-500/40';
        verdict.innerHTML = `<div class="text-emerald-400 font-bold text-lg">✅ HAM — LEGITIMATE EMAIL (${pct} Spam Risk)</div>
          <p class="text-xs text-slate-300 mt-1">Verified safe by <strong>${data.model_used}</strong>. Kept cleanly in Inbox.</p>`;
      }

      document.getElementById('sandbox-consensus').innerHTML = data.consensus.map(c =>
        `<div class="flex justify-between"><span>${c.model}</span><span class="${c.prediction === 'SPAM' ? 'text-red-400 font-semibold' : 'text-emerald-400'}">${c.prediction} (${(c.spam_probability*100).toFixed(1)}%)</span></div>`
      ).join('');

      document.getElementById('sandbox-tokens').innerHTML = data.top_tokens.map(t =>
        `<span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono">${t.token}: ${t.score}</span>`
      ).join('');
    }

    function renderScanRows(logs) {
      const tbody = document.getElementById('scan-logs-body');
      if (!logs || logs.length === 0) return;
      tbody.innerHTML = logs.map(r => `
        <tr>
          <td class="py-2.5 pr-3 font-semibold ${r.prediction === 'SPAM' ? 'text-red-400' : 'text-emerald-400'}">
            ${r.prediction === 'SPAM' ? '🚨 SPAM' : '✅ HAM'}
          </td>
          <td class="py-2.5 pr-3 text-slate-300">${(r.spam_probability * 100).toFixed(1)}%</td>
          <td class="py-2.5 pr-3 text-slate-300 max-w-[160px] truncate">${r.from || ''}</td>
          <td class="py-2.5 pr-3 text-white max-w-[220px] truncate">${r.subject || ''}</td>
          <td class="py-2.5 text-slate-400">${r.action_taken || ''}</td>
        </tr>
      `).join('');
    }

    async function saveUserSettings() {
      const payload = {
        label_name: document.getElementById('setting-label').value,
        auto_filter_enabled: document.getElementById('setting-auto').checked,
        move_to_spam: document.getElementById('setting-move').checked
      };
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      if (res.ok) showAlert('Saved your Gmail & Android filter preferences!');
    }

    async function scanMyInbox() {
      const btn = document.getElementById('btn-scan-now');
      btn.disabled = true;
      btn.textContent = '⏳ Scanning Gmail...';
      try {
        const res = await fetch('/api/gmail/scan', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({max_emails: 15})
        });
        const data = await res.json();
        if (!res.ok) {
          showAlert(data.detail || 'Scan failed', true);
        } else {
          showAlert(`Scanned ${data.scanned_count} recent emails — flagged ${data.spam_flagged} as spam! Check your Android Gmail app.`);
          document.getElementById('stat-scanned').textContent = data.user_stats.total_scanned;
          document.getElementById('stat-spam').textContent = data.user_stats.spam_detected;
          document.getElementById('stat-last-scan').textContent = data.user_stats.last_scanned_at;
          renderScanRows(data.results);
        }
      } finally {
        btn.disabled = false;
        btn.textContent = '🔄 Scan My Gmail Inbox Now';
      }
    }

    async function logoutUser() {
      await fetch('/api/auth/logout', {method: 'POST'});
      window.location.href = '/';
    }

    async function initDashboard() {
      const params = new URLSearchParams(window.location.search);
      if (params.get('error')) showAlert(params.get('error'), true);
      if (params.get('connected')) showAlert('Connected your Google Account! Your Gmail & Android auto-filter is ready.');

      const res = await fetch('/api/me');
      if (!res.ok) return;
      const data = await res.json();

      document.getElementById('oauth-redirect-code').textContent = data.oauth_redirect_uri;
      document.getElementById('sandbox-model-badge').textContent = `Active Model: ${data.best_model_name}`;

      if (!data.oauth_configured) {
        document.getElementById('setup-guide').classList.remove('hidden');
      }

      if (data.authenticated && data.user) {
        const u = data.user;
        document.getElementById('user-dashboard').classList.remove('hidden');
        document.getElementById('stat-scanned').textContent = u.total_scanned;
        document.getElementById('stat-spam').textContent = u.spam_detected;
        document.getElementById('stat-last-scan').textContent = u.last_scanned_at;
        document.getElementById('setting-label').value = u.label_name;
        document.getElementById('setting-auto').checked = u.auto_filter_enabled;
        document.getElementById('setting-move').checked = u.move_to_spam;
        renderScanRows(u.recent_logs);

        document.getElementById('nav-auth-area').innerHTML = `
          <div class="flex items-center gap-3">
            <span class="text-xs text-slate-300 hidden sm:inline">Connected: <strong>${u.email}</strong></span>
            <button onclick="logoutUser()" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">
              Sign Out
            </button>
          </div>
        `;
      }
    }

    initDashboard();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
@app.get("/api/index", response_class=HTMLResponse, include_in_schema=False)
@app.get("/api/index.py", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    return HTMLResponse(content=FRONTEND_HTML)
