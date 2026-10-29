"""
Gmail & Android Gmail App Auto-Filter Integration (IMAP over SSL)
Automatically scans incoming Gmail messages, classifies them using the trained
ML model, and applies a custom Gmail label (default: 'AI-Spam') while keeping
them safely in the Inbox. All label changes sync immediately to the Gmail Android app.
"""

import argparse
import email
from email.header import decode_header
from email.message import Message
import imaplib
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
from dotenv import load_dotenv

from preprocessing import preprocess_text
from train_model import train_and_evaluate

load_dotenv()

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "models", "spam_artifacts.joblib")
PROCESSED_LOG_PATH = os.path.join(os.path.dirname(__file__), "data", "processed_emails.json")


def decode_mime_words(header_val: Optional[str]) -> str:
    """Decode RFC-2047 encoded email headers (Subject, From) into clean UTF-8 strings."""
    if not header_val:
        return ""
    decoded_parts = []
    for part, encoding in decode_header(header_val):
        if isinstance(part, bytes):
            charset = encoding or "utf-8"
            try:
                decoded_parts.append(part.decode(charset, errors="replace"))
            except LookupError:
                decoded_parts.append(part.decode("utf-8", errors="replace"))
        else:
            decoded_parts.append(str(part))
    return "".join(decoded_parts).strip()


def extract_email_body(msg: Message) -> str:
    """Extract plain text (or stripped HTML fallback) from a MIME email message."""
    plain_parts: List[str] = []
    html_parts: List[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disp = str(part.get("Content-Disposition") or "")
            if "attachment" in content_disp.lower():
                continue

            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            charset = part.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except LookupError:
                text = payload.decode("utf-8", errors="replace")

            if content_type == "text/plain":
                plain_parts.append(text)
            elif content_type == "text/html":
                html_parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                text = payload.decode(charset, errors="replace")
            except LookupError:
                text = payload.decode("utf-8", errors="replace")
            if msg.get_content_type() == "text/html":
                html_parts.append(text)
            else:
                plain_parts.append(text)

    if plain_parts:
        return "\n".join(plain_parts).strip()

    if html_parts:
        raw_html = "\n".join(html_parts)
        raw_html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", raw_html)
        return re.sub(r"\s+", " ", clean).strip()

    return ""


def load_processed_cache() -> Dict[str, dict]:
    """Load record of already-scanned Gmail UIDs so we don't re-process messages."""
    if os.path.exists(PROCESSED_LOG_PATH):
        try:
            with open(PROCESSED_LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_processed_cache(cache: Dict[str, dict]) -> None:
    """Persist scanned email history to disk."""
    os.makedirs(os.path.dirname(PROCESSED_LOG_PATH), exist_ok=True)
    with open(PROCESSED_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def ensure_gmail_label(imap: imaplib.IMAP4_SSL, label_name: str) -> None:
    """Create the custom Gmail label if it does not already exist."""
    try:
        imap.create(f'"{label_name}"')
    except imaplib.IMAP4.error:
        # Label already exists on Gmail server
        pass


def apply_gmail_label(
    imap: imaplib.IMAP4_SSL,
    uid: bytes,
    label_name: str,
    move_to_spam: bool = False,
) -> bool:
    """
    Apply a custom Gmail label (e.g. 'AI-Spam') to a message UID without
    marking it as read or removing it from INBOX (unless move_to_spam=True).
    """
    uid_str = uid.decode("utf-8") if isinstance(uid, bytes) else str(uid)
    ensure_gmail_label(imap, label_name)

    labeled_ok = False
    # 1. Use Gmail's native X-GM-LABELS extension
    try:
        typ, _ = imap.uid("STORE", uid_str, "+X-GM-LABELS", f'("{label_name}")')
        if typ == "OK":
            labeled_ok = True
    except Exception:
        pass

    # 2. Also COPY to the label mailbox (standard IMAP way to attach a Gmail label)
    if not labeled_ok:
        try:
            typ, _ = imap.uid("COPY", uid_str, f'"{label_name}"')
            if typ == "OK":
                labeled_ok = True
        except Exception:
            pass

    # Optional: Move out of Inbox into [Gmail]/Spam only if explicitly requested
    if move_to_spam:
        try:
            imap.uid("COPY", uid_str, '"[Gmail]/Spam"')
            imap.uid("STORE", uid_str, "+FLAGS", "(\\Deleted)")
            imap.expunge()
        except Exception:
            pass

    return labeled_ok


def scan_gmail_inbox(
    email_address: str,
    app_password: str,
    label_name: str = "AI-Spam",
    max_emails: int = 15,
    unread_only: bool = False,
    move_to_spam: bool = False,
    model_name: Optional[str] = None,
    force_rescan: bool = False,
) -> Tuple[List[dict], Optional[str]]:
    """
    Connect to Gmail via IMAP, classify recent emails using the trained model,
    and apply `label_name` to any detected spam while keeping it in Inbox.

    Returns (results_list, error_message_if_any).
    """
    if not email_address or not app_password:
        return [], "Missing Gmail address or 16-character Google App Password."

    # Clean spaces from 16-character Google App Password (e.g. "abcd efgh ijkl mnop")
    clean_password = app_password.replace(" ", "").strip()
    clean_email = email_address.strip()

    if not os.path.exists(ARTIFACTS_PATH):
        artifacts = train_and_evaluate()
    else:
        artifacts = joblib.load(ARTIFACTS_PATH)

    vectorizer = artifacts["vectorizer"]
    chosen_model_name = model_name or artifacts["best_model_name"]
    model = artifacts["models"][chosen_model_name]

    processed_cache = load_processed_cache()
    results: List[dict] = []

    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        imap.login(clean_email, clean_password)
        imap.select("INBOX", readonly=False)

        search_criterion = "UNSEEN" if unread_only else "ALL"
        typ, data = imap.uid("SEARCH", None, search_criterion)
        if typ != "OK" or not data or not data[0]:
            imap.logout()
            return [], None

        all_uids = data[0].split()
        recent_uids = list(reversed(all_uids[-max_emails:]))

        for uid in recent_uids:
            uid_str = uid.decode("utf-8")

            if not force_rescan and uid_str in processed_cache:
                results.append(processed_cache[uid_str])
                continue

            # Use BODY.PEEK[] so unread emails stay unread in Gmail / Android notifications
            typ, msg_data = imap.uid("FETCH", uid_str, "(BODY.PEEK[])")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue

            raw_bytes = msg_data[0][1]
            if not isinstance(raw_bytes, bytes):
                continue

            msg = email.message_from_bytes(raw_bytes)
            subject = decode_mime_words(msg.get("Subject", "(No Subject)"))
            sender = decode_mime_words(msg.get("From", "(Unknown Sender)"))
            date_hdr = decode_mime_words(msg.get("Date", ""))
            body = extract_email_body(msg)

            combined_text = f"{subject}\n{body}".strip()
            clean_text = preprocess_text(combined_text)
            X_vec = vectorizer.transform([clean_text])

            spam_prob = float(model.predict_proba(X_vec)[0][1])
            pred = 1 if spam_prob > 0.86 else 0

            action_taken = "Kept in Inbox (Ham)"
            if pred == 1:
                labeled = apply_gmail_label(
                    imap, uid, label_name=label_name, move_to_spam=move_to_spam
                )
                if move_to_spam:
                    action_taken = f"Labeled '{label_name}' & Moved to Spam"
                elif labeled:
                    action_taken = f"Labeled '{label_name}' (Kept in Inbox)"
                else:
                    action_taken = "Flagged as Spam (Label pending)"

            record = {
                "uid": uid_str,
                "from": sender,
                "subject": subject,
                "date": date_hdr,
                "snippet": (body[:140] + "...") if len(body) > 140 else body,
                "prediction": "SPAM" if pred == 1 else "HAM",
                "spam_probability": round(spam_prob, 4),
                "model_used": chosen_model_name,
                "action_taken": action_taken,
                "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            processed_cache[uid_str] = record
            results.append(record)

        save_processed_cache(processed_cache)
        imap.close()
        imap.logout()
        return results, None

    except imaplib.IMAP4.error as auth_err:
        return (
            [],
            f"Gmail IMAP Authentication Error: {auth_err}. "
            "Make sure 2-Step Verification is enabled on your Google Account and you are using "
            "a 16-character Google App Password (not your regular Gmail password).",
        )
    except Exception as exc:
        return [], f"Connection error: {exc}"


def run_daemon(
    email_address: str,
    app_password: str,
    label_name: str = "AI-Spam",
    poll_interval: int = 60,
    max_emails: int = 15,
    unread_only: bool = False,
    move_to_spam: bool = False,
) -> None:
    """Continuously watch Gmail Inbox and label new spam emails automatically."""
    print("=" * 72)
    print("📲 Gmail & Android Auto-Filter Daemon Started")
    print(f"   Account       : {email_address}")
    print(f"   Target Label  : '{label_name}' (syncs to Gmail Android App)")
    print(f"   Mode          : {'Move to Spam' if move_to_spam else 'Keep in Inbox + Add Label'}")
    print(f"   Poll Interval : Every {poll_interval} seconds")
    print("=" * 72)

    while True:
        results, err = scan_gmail_inbox(
            email_address=email_address,
            app_password=app_password,
            label_name=label_name,
            max_emails=max_emails,
            unread_only=unread_only,
            move_to_spam=move_to_spam,
        )
        now = datetime.now().strftime("%H:%M:%S")
        if err:
            print(f"[{now}] ⚠️ Error: {err}")
        else:
            spam_hits = [r for r in results if r["prediction"] == "SPAM"]
            print(
                f"[{now}] ✅ Checked {len(results)} recent messages — "
                f"{len(spam_hits)} flagged with label '{label_name}'."
            )
            for hit in spam_hits:
                print(
                    f"   🚨 [{hit['spam_probability']:.1%}] {hit['subject'][:60]} "
                    f"(From: {hit['from'][:40]}) -> {hit['action_taken']}"
                )
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Automatically scan Gmail inbox and label spam for Gmail Web & Android App."
    )
    parser.add_argument("--email", default=os.getenv("GMAIL_ADDRESS", ""), help="Your Gmail address")
    parser.add_argument(
        "--app-password",
        default=os.getenv("GMAIL_APP_PASSWORD", ""),
        help="Your 16-character Google App Password",
    )
    parser.add_argument(
        "--label",
        default=os.getenv("GMAIL_SPAM_LABEL", "AI-Spam"),
        help="Custom Gmail label to attach to detected spam (default: AI-Spam)",
    )
    parser.add_argument(
        "--max-emails",
        type=int,
        default=15,
        help="Number of most recent inbox emails to check per scan (default: 15)",
    )
    parser.add_argument(
        "--unread-only",
        action="store_true",
        help="Only scan unread (UNSEEN) emails",
    )
    parser.add_argument(
        "--move-to-spam",
        action="store_true",
        help="Move flagged emails directly to [Gmail]/Spam instead of keeping in Inbox",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit instead of continuous background polling",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scan emails even if they were already scanned previously",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds for continuous background mode (default: 60)",
    )

    args = parser.parse_args()

    if not args.email or not args.app_password:
        print(
            "⚠️ Missing credentials! Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in your .env file "
            "or pass --email and --app-password."
        )
        return

    if args.once:
        print(f"🔍 Scanning top {args.max_emails} emails in {args.email}...")
        results, err = scan_gmail_inbox(
            email_address=args.email,
            app_password=args.app_password,
            label_name=args.label,
            max_emails=args.max_emails,
            unread_only=args.unread_only,
            move_to_spam=args.move_to_spam,
            force_rescan=args.force,
        )
        if err:
            print(f"❌ Error: {err}")
            return
        for r in results:
            badge = "🚨 SPAM" if r["prediction"] == "SPAM" else "✅ HAM "
            print(
                f"{badge} ({r['spam_probability']:.1%}) | {r['subject'][:55]:<55} | {r['action_taken']}"
            )
    else:
        run_daemon(
            email_address=args.email,
            app_password=args.app_password,
            label_name=args.label,
            poll_interval=args.interval,
            max_emails=args.max_emails,
            unread_only=args.unread_only,
            move_to_spam=args.move_to_spam,
        )


if __name__ == "__main__":
    main()
