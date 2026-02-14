import base64
import hashlib
import hmac
import json
import os
import random
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib import parse, request

from flask import Flask, jsonify, redirect, render_template, request as flask_request

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "8080")))
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/revenue_bot.db")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
AUTO_GENERATE_INTERVAL_MINUTES = int(os.getenv("AUTO_GENERATE_INTERVAL_MINUTES", "60"))
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
APP_PUBLIC_URL = os.getenv("APP_PUBLIC_URL", "")
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox")
PAYOUT_SENDER_EMAIL = os.getenv("PAYOUT_SENDER_EMAIL", "")

PAYPAL_BASE = "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"

app = Flask(__name__, template_folder="../templates")


DIGITAL_PRODUCT_BLUEPRINTS = [
    "Notion productivity template pack",
    "Resume and cover letter kit",
    "Small business invoice template bundle",
    "Meal prep planner bundle",
    "Freelancer outreach email pack",
    "Social media caption template pack",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            price_cents INTEGER NOT NULL,
            checkout_url TEXT NOT NULL,
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sales (
            id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            currency TEXT NOT NULL,
            buyer_venmo_handle TEXT,
            provider TEXT NOT NULL,
            provider_event_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS payouts (
            id TEXT PRIMARY KEY,
            sale_id TEXT NOT NULL,
            venmo_recipient TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            status TEXT NOT NULL,
            provider_batch_id TEXT,
            note TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def create_product() -> dict:
    title = random.choice(DIGITAL_PRODUCT_BLUEPRINTS)
    price_cents = random.choice([900, 1200, 1500, 1900, 2500])
    product_id = str(uuid.uuid4())

    checkout_url = f"/checkout/{product_id}"

    conn = db()
    conn.execute(
        "INSERT INTO products (id, title, price_cents, checkout_url, created_at, active) VALUES (?, ?, ?, ?, ?, 1)",
        (product_id, title, price_cents, checkout_url, utc_now_iso()),
    )
    conn.commit()
    conn.close()
    return {
        "id": product_id,
        "title": title,
        "price_cents": price_cents,
        "checkout_url": checkout_url,
    }


def list_products(active_only: bool = True) -> list[dict]:
    conn = db()
    if active_only:
        rows = conn.execute(
            "SELECT id, title, price_cents, checkout_url, created_at FROM products WHERE active = 1 ORDER BY created_at DESC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, title, price_cents, checkout_url, created_at, active FROM products ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_product(product_id: str) -> dict | None:
    conn = db()
    row = conn.execute(
        "SELECT id, title, price_cents, checkout_url, created_at FROM products WHERE id = ? AND active = 1",
        (product_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def public_base_url() -> str:
    configured = APP_PUBLIC_URL.strip()
    if configured:
        return configured.rstrip("/")
    return flask_request.url_root.rstrip("/")


def create_stripe_checkout_session(product: dict, venmo_handle: str) -> str:
    if not STRIPE_SECRET_KEY:
        raise ValueError("missing STRIPE_SECRET_KEY")

    base_url = public_base_url()
    payload: list[tuple[str, str]] = [
        ("mode", "payment"),
        ("success_url", f"{base_url}/?checkout=success"),
        ("cancel_url", f"{base_url}/?checkout=cancel"),
        ("line_items[0][price_data][currency]", "usd"),
        ("line_items[0][price_data][unit_amount]", str(product["price_cents"])),
        ("line_items[0][price_data][product_data][name]", product["title"]),
        ("line_items[0][quantity]", "1"),
        ("metadata[product_id]", product["id"]),
    ]

    if venmo_handle:
        payload.append(("metadata[venmo_handle]", venmo_handle))

    body = parse.urlencode(payload).encode("utf-8")
    req = request.Request(
        "https://api.stripe.com/v1/checkout/sessions",
        data=body,
        headers={
            "Authorization": f"Bearer {STRIPE_SECRET_KEY}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        session_url = data.get("url", "")
        if not session_url:
            raise ValueError("stripe response missing session URL")
        return session_url


def upsert_sale_and_payout(event: dict) -> tuple[str, str]:
    sale_id = str(uuid.uuid4())
    payout_id = str(uuid.uuid4())

    data_object = event.get("data", {}).get("object", {})
    product_id = data_object.get("metadata", {}).get("product_id") or "unknown"
    venmo_handle = data_object.get("metadata", {}).get("venmo_handle") or ""
    amount_cents = int(data_object.get("amount_total") or 0)
    currency = (data_object.get("currency") or "usd").upper()
    provider_event_id = event.get("id", "")

    conn = db()
    conn.execute(
        "INSERT INTO sales (id, product_id, amount_cents, currency, buyer_venmo_handle, provider, provider_event_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (sale_id, product_id, amount_cents, currency, venmo_handle, "stripe", provider_event_id, utc_now_iso()),
    )
    conn.execute(
        "INSERT INTO payouts (id, sale_id, venmo_recipient, amount_cents, status, provider_batch_id, note, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (payout_id, sale_id, venmo_handle, amount_cents, "pending", None, "queued", utc_now_iso()),
    )
    conn.commit()
    conn.close()
    return sale_id, payout_id


def verify_stripe_signature(raw_body: bytes, header: str) -> bool:
    if not STRIPE_WEBHOOK_SECRET:
        return False
    if not header:
        return False

    parts = header.split(",")
    ts = ""
    signature = ""
    for p in parts:
        key, _, value = p.partition("=")
        if key == "t":
            ts = value
        if key == "v1":
            signature = value

    if not ts or not signature:
        return False

    payload = f"{ts}.".encode("utf-8") + raw_body
    expected = hmac.new(STRIPE_WEBHOOK_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def paypal_access_token() -> str:
    creds = f"{PAYPAL_CLIENT_ID}:{PAYPAL_CLIENT_SECRET}".encode("utf-8")
    auth_header = base64.b64encode(creds).decode("utf-8")

    req = request.Request(
        f"{PAYPAL_BASE}/v1/oauth2/token",
        data=parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8"),
        headers={
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
        return payload["access_token"]


def send_venmo_payout(recipient: str, amount_cents: int) -> tuple[bool, str, str]:
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        return True, "simulated", "missing PayPal credentials; marked as simulated"
    if not recipient:
        return False, "", "missing Venmo recipient"

    token = paypal_access_token()
    batch_id = f"batch_{uuid.uuid4()}"
    amount_str = f"{amount_cents / 100:.2f}"

    payload = {
        "sender_batch_header": {
            "sender_batch_id": batch_id,
            "email_subject": "You have a payout",
            "email_message": "Automated payout from revenue bot",
        },
        "items": [
            {
                "recipient_type": "EMAIL",
                "amount": {"value": amount_str, "currency": "USD"},
                "receiver": recipient,
                "note": "Automated payout",
                "sender_item_id": str(uuid.uuid4()),
                "recipient_wallet": "VENMO",
            }
        ],
    }

    req = request.Request(
        f"{PAYPAL_BASE}/v1/payments/payouts",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
            provider_batch_id = response_data.get("batch_header", {}).get("payout_batch_id", batch_id)
            return True, provider_batch_id, "submitted"
    except Exception as exc:
        return False, "", str(exc)


def process_pending_payouts(limit: int = 10) -> dict:
    conn = db()
    rows = conn.execute(
        "SELECT id, venmo_recipient, amount_cents FROM payouts WHERE status = 'pending' ORDER BY updated_at ASC LIMIT ?",
        (limit,),
    ).fetchall()

    processed = 0
    failed = 0

    for row in rows:
        payout_id = row["id"]
        ok, batch_id, note = send_venmo_payout(row["venmo_recipient"], row["amount_cents"])
        new_status = "paid" if ok else "failed"
        if ok:
            processed += 1
        else:
            failed += 1

        conn.execute(
            "UPDATE payouts SET status = ?, provider_batch_id = ?, note = ?, updated_at = ? WHERE id = ?",
            (new_status, batch_id, note, utc_now_iso(), payout_id),
        )

    conn.commit()
    conn.close()
    return {"processed": processed, "failed": failed}


def admin_guard() -> bool:
    return flask_request.headers.get("x-admin-token") == ADMIN_TOKEN


@app.get("/")
def dashboard():
    products = list_products()
    conn = db()
    payout_rows = conn.execute(
        "SELECT id, venmo_recipient, amount_cents, status, note, updated_at FROM payouts ORDER BY updated_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    payouts = [dict(r) for r in payout_rows]
    return render_template("index.html", products=products, payouts=payouts)


@app.get("/health")
def health():
    return jsonify({"ok": True, "time": utc_now_iso()})


@app.get("/api/products")
def api_products():
    return jsonify({"products": list_products()})


@app.get("/checkout/<product_id>")
def checkout(product_id: str):
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404

    venmo_handle = (flask_request.args.get("venmo_handle") or "").strip()
    try:
        session_url = create_stripe_checkout_session(product, venmo_handle)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": "stripe api error", "detail": detail}), 502
    except URLError as exc:
        return jsonify({"error": "stripe network error", "detail": str(exc)}), 502

    return redirect(session_url, code=302)


@app.post("/admin/generate")
def admin_generate():
    if not admin_guard():
        return jsonify({"error": "unauthorized"}), 401

    product = create_product()
    return jsonify({"product": product})


@app.post("/admin/run-payouts")
def admin_run_payouts():
    if not admin_guard():
        return jsonify({"error": "unauthorized"}), 401

    result = process_pending_payouts(limit=25)
    return jsonify(result)


@app.post("/webhooks/stripe")
def stripe_webhook():
    raw = flask_request.get_data(cache=False)
    signature = flask_request.headers.get("Stripe-Signature", "")

    if not verify_stripe_signature(raw, signature):
        return jsonify({"error": "invalid signature"}), 400

    event = flask_request.get_json(silent=True) or {}
    if event.get("type") != "checkout.session.completed":
        return jsonify({"received": True, "ignored": True})

    sale_id, payout_id = upsert_sale_and_payout(event)
    return jsonify({"received": True, "sale_id": sale_id, "payout_id": payout_id})


def auto_generator_loop(stop_event: threading.Event) -> None:
    interval_seconds = max(5, AUTO_GENERATE_INTERVAL_MINUTES * 60)
    while not stop_event.is_set():
        try:
            create_product()
        except Exception:
            pass
        stop_event.wait(interval_seconds)


def main() -> None:
    init_db()

    stop_event = threading.Event()
    generator_thread = threading.Thread(target=auto_generator_loop, args=(stop_event,), daemon=True)
    generator_thread.start()

    app.run(host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    main()
