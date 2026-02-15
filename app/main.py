import base64
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from urllib import parse, request
from urllib.error import HTTPError, URLError

from flask import Flask, jsonify, redirect, render_template, request as flask_request

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "8080")))
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/revenue_bot.db")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
AUTO_GENERATE_INTERVAL_MINUTES = int(os.getenv("AUTO_GENERATE_INTERVAL_MINUTES", "60"))
MIN_STORE_PRODUCTS = int(os.getenv("MIN_STORE_PRODUCTS", "12"))
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
APP_PUBLIC_URL = os.getenv("APP_PUBLIC_URL", "")
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox")
PAYOUT_SENDER_EMAIL = os.getenv("PAYOUT_SENDER_EMAIL", "")

PAYPAL_BASE = "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"

app = Flask(__name__, template_folder="../templates")


CATALOG_PRODUCTS = [
    {
        "title": "Creator Caption Vault",
        "category": "Creator Growth",
        "price_cents": 1900,
        "tagline": "Write faster with high-converting caption frameworks.",
        "description": "A premium caption system for creators and coaches who want consistent posting without content fatigue.",
        "preview_items": [
            "120 caption starters by goal (sales, engagement, authority)",
            "Hook bank for short-form posts",
            "CTA matrix for comments, DMs, and link clicks",
        ],
        "preview_snippet": "You do not need more content ideas. You need better structure. Start with a sharp hook, add one clear insight, then close with a single action readers can take today.",
    },
    {
        "title": "Short-Form Hook Library",
        "category": "Creator Growth",
        "price_cents": 1700,
        "tagline": "Hooks engineered for retention in the first 3 seconds.",
        "description": "A swipe file of high-performing hooks tailored for Instagram Reels, TikTok, and YouTube Shorts.",
        "preview_items": [
            "200 short-form opening lines",
            "Pattern interrupt formulas",
            "Hook variations by audience awareness level",
        ],
        "preview_snippet": "If your content is good but views are flat, your opener is the bottleneck. This line works because it calls out a problem your audience already feels in their day-to-day workflow.",
    },
    {
        "title": "UGC Pitch Deck Kit",
        "category": "Creator Business",
        "price_cents": 2900,
        "tagline": "Pitch brands with a polished, trust-building media deck.",
        "description": "A complete Canva deck template for UGC creators, including service pages, pricing options, and deliverables.",
        "preview_items": [
            "12-slide brand pitch deck",
            "Rate card and package layout",
            "Case-study and testimonial slide templates",
        ],
        "preview_snippet": "I help consumer brands increase ad performance with authentic short-form creative. My packages are designed for testing velocity, not one-off vanity content.",
    },
    {
        "title": "Notion Operator System",
        "category": "Productivity",
        "price_cents": 2400,
        "tagline": "One workspace for planning, execution, and weekly review.",
        "description": "A serious operating system for solopreneurs balancing content, clients, and internal projects.",
        "preview_items": [
            "Daily command center",
            "Project and task pipeline",
            "Weekly scorecard and review dashboard",
        ],
        "preview_snippet": "This week, focus on three outcomes only. A smaller, priority-driven plan with daily execution blocks consistently outperforms long task lists.",
    },
    {
        "title": "Budget & Cashflow Spreadsheet Pro",
        "category": "Finance",
        "price_cents": 2100,
        "tagline": "Track spending, forecast cashflow, and plan profit.",
        "description": "A practical spreadsheet for freelancers and households that need monthly clarity without accounting complexity.",
        "preview_items": [
            "Monthly budget dashboard",
            "Cashflow forecast by category",
            "Debt payoff and savings tracker",
        ],
        "preview_snippet": "Revenue can look healthy while cashflow is tight. Forecasting the next 8 weeks gives early warning on expenses and helps you avoid reactive decisions.",
    },
    {
        "title": "Resume + Interview Kit",
        "category": "Career",
        "price_cents": 2300,
        "tagline": "Position your experience around measurable impact.",
        "description": "A modern job-search package with ATS-friendly resume templates and interview preparation assets.",
        "preview_items": [
            "ATS resume template set",
            "Cover letter framework",
            "Interview answer bank for common questions",
        ],
        "preview_snippet": "Hiring managers scan for outcomes first. Lead with quantified wins and make your first three bullet points impossible to ignore.",
    },
    {
        "title": "Freelance Client Pack",
        "category": "Freelance Ops",
        "price_cents": 2600,
        "tagline": "Contracts, onboarding, and invoices in one client-ready bundle.",
        "description": "A plug-and-play document suite for freelancers who want smoother onboarding and fewer payment delays.",
        "preview_items": [
            "Service agreement template",
            "Client onboarding questionnaire",
            "Invoice and late-fee policy template",
        ],
        "preview_snippet": "Clear scope and payment terms reduce revision disputes. This template language is designed to set expectations before the project starts.",
    },
    {
        "title": "Wedding Invite Suite (Canva)",
        "category": "Events",
        "price_cents": 2700,
        "tagline": "Elegant invitation system for modern weddings.",
        "description": "A coordinated stationery suite with invitation, RSVP, details card, and day-of signage templates.",
        "preview_items": [
            "Invitation + RSVP templates",
            "Timeline/details card",
            "Welcome sign and table number set",
        ],
        "preview_snippet": "Join us for a celebration of love, laughter, and forever. Designed with clean typography and timeless layout for easy customization.",
    },
    {
        "title": "Airbnb Host Welcome Book",
        "category": "Hospitality",
        "price_cents": 1800,
        "tagline": "Reduce guest questions with a polished digital house guide.",
        "description": "A modern welcome guide for short-term rental hosts to improve guest experience and reduce repetitive support messages.",
        "preview_items": [
            "House rules and quick-start page",
            "Local recommendations layout",
            "Checkout checklist template",
        ],
        "preview_snippet": "Welcome to your stay. This guide covers everything from Wi-Fi and parking to local food spots and checkout steps so your trip runs smoothly.",
    },
    {
        "title": "Etsy Listing SEO Toolkit",
        "category": "Ecommerce",
        "price_cents": 2200,
        "tagline": "Improve discoverability with search-driven listing templates.",
        "description": "A toolkit for Etsy sellers to structure titles, tags, and product descriptions around buyer intent.",
        "preview_items": [
            "Listing title formula library",
            "Tag planner worksheet",
            "Photo and thumbnail optimization checklist",
        ],
        "preview_snippet": "The best-performing listings prioritize intent clarity over clever wording. Buyers should understand exactly what they are purchasing at a glance.",
    },
    {
        "title": "Meal Prep Planner + Grocery System",
        "category": "Wellness",
        "price_cents": 1600,
        "tagline": "Plan meals once and simplify the entire week.",
        "description": "A practical nutrition planning bundle for busy professionals and families.",
        "preview_items": [
            "7-day meal planner",
            "Grocery list generator layout",
            "Batch prep workflow sheet",
        ],
        "preview_snippet": "Plan protein first, then build repeatable lunches and dinners around it. This method cuts prep time and avoids midweek decision fatigue.",
    },
    {
        "title": "Kids Chore & Reward Chart Pack",
        "category": "Family",
        "price_cents": 1400,
        "tagline": "Make routines easier with visual habit trackers.",
        "description": "Printable and editable charts for parents building consistency around chores and daily responsibilities.",
        "preview_items": [
            "Morning and evening routine charts",
            "Weekly chore tracker",
            "Reward milestone board",
        ],
        "preview_snippet": "Children follow routines more consistently when expectations are visible. Keep goals simple, celebrate streaks, and reward completion milestones.",
    },
    {
        "title": "SOP Manual for Small Teams",
        "category": "Operations",
        "price_cents": 3200,
        "tagline": "Document recurring processes without overcomplicating operations.",
        "description": "A standard operating procedure template system for startups and small agencies.",
        "preview_items": [
            "SOP index and ownership map",
            "Process template with QA checklist",
            "Change log and version control page",
        ],
        "preview_snippet": "Good SOPs are specific enough to execute and simple enough to maintain. This layout keeps process knowledge usable as your team grows.",
    },
    {
        "title": "Course Launch Planner",
        "category": "Creator Business",
        "price_cents": 2500,
        "tagline": "Launch with a clear timeline, messaging plan, and KPI tracker.",
        "description": "A campaign planner for educators and creators building and launching digital courses.",
        "preview_items": [
            "Pre-launch timeline",
            "Email and social promo calendar",
            "Launch day KPI dashboard",
        ],
        "preview_snippet": "A strong launch is operational, not chaotic. Mapping content, email, and offer deadlines in one timeline improves execution and conversion quality.",
    },
    {
        "title": "Brand Kit + Social Template Bundle",
        "category": "Branding",
        "price_cents": 2800,
        "tagline": "Build a consistent visual identity across all channels.",
        "description": "A full brand starter package with logo lockups, color system guidance, and social post templates.",
        "preview_items": [
            "Brand style guide template",
            "Instagram post and story layouts",
            "Launch announcement templates",
        ],
        "preview_snippet": "Consistency drives trust. Use one visual language across posts, sales pages, and client touchpoints so your brand feels instantly recognizable.",
    },
    {
        "title": "Real Estate Lead Magnet Pack",
        "category": "Real Estate",
        "price_cents": 2600,
        "tagline": "Capture and nurture buyer and seller leads faster.",
        "description": "A lead generation set for real estate agents with downloadable guides and follow-up sequences.",
        "preview_items": [
            "Homebuyer guide template",
            "Seller prep checklist",
            "Lead follow-up email scripts",
        ],
        "preview_snippet": "Most leads are not ready on day one. This sequence helps you build trust over time with value-first follow-up content.",
    },
]

CATALOG_BY_TITLE = {item["title"]: item for item in CATALOG_PRODUCTS}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def enrich_product(product: dict) -> dict:
    details = CATALOG_BY_TITLE.get(product.get("title", ""), {})
    product["category"] = details.get("category", "General")
    product["tagline"] = details.get("tagline", "High-value digital toolkit.")
    product["description"] = details.get(
        "description",
        "A practical digital product designed to save time and improve outcomes.",
    )
    product["preview_items"] = details.get(
        "preview_items",
        ["Editable files", "Step-by-step guide", "Instant digital delivery"],
    )
    product["preview_snippet"] = details.get(
        "preview_snippet",
        "Sample preview will be provided after purchase.",
    )
    return product


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
    return [enrich_product(dict(r)) for r in rows]


def get_product(product_id: str) -> dict | None:
    conn = db()
    row = conn.execute(
        "SELECT id, title, price_cents, checkout_url, created_at FROM products WHERE id = ? AND active = 1",
        (product_id,),
    ).fetchone()
    conn.close()
    return enrich_product(dict(row)) if row else None


def get_product_by_title(title: str) -> dict | None:
    conn = db()
    row = conn.execute(
        "SELECT id, title, price_cents, checkout_url, created_at FROM products WHERE title = ? AND active = 1 ORDER BY created_at DESC LIMIT 1",
        (title,),
    ).fetchone()
    conn.close()
    return enrich_product(dict(row)) if row else None


def count_active_products() -> int:
    conn = db()
    row = conn.execute("SELECT COUNT(*) AS c FROM products WHERE active = 1").fetchone()
    conn.close()
    return int(row["c"]) if row else 0


def deactivate_duplicate_products() -> int:
    conn = db()
    rows = conn.execute(
        "SELECT id, title FROM products WHERE active = 1 ORDER BY title ASC, created_at DESC"
    ).fetchall()
    seen_titles: set[str] = set()
    duplicates: list[str] = []

    for row in rows:
        product_id = row["id"]
        title = row["title"]
        if title in seen_titles:
            duplicates.append(product_id)
        else:
            seen_titles.add(title)

    for product_id in duplicates:
        conn.execute("UPDATE products SET active = 0 WHERE id = ?", (product_id,))

    conn.commit()
    conn.close()
    return len(duplicates)


def next_missing_catalog_item() -> dict | None:
    active_titles = {p["title"] for p in list_products(active_only=True)}
    for item in CATALOG_PRODUCTS:
        if item["title"] not in active_titles:
            return item
    return None


def create_product(item: dict | None = None) -> dict:
    seed = item or next_missing_catalog_item()
    if not seed:
        products = list_products(active_only=True)
        if products:
            return products[0]
        seed = CATALOG_PRODUCTS[0]

    existing = get_product_by_title(seed["title"])
    if existing:
        return existing

    product_id = str(uuid.uuid4())
    checkout_url = f"/checkout/{product_id}"

    conn = db()
    conn.execute(
        "INSERT INTO products (id, title, price_cents, checkout_url, created_at, active) VALUES (?, ?, ?, ?, ?, 1)",
        (product_id, seed["title"], int(seed["price_cents"]), checkout_url, utc_now_iso()),
    )
    conn.commit()
    conn.close()

    return enrich_product(
        {
            "id": product_id,
            "title": seed["title"],
            "price_cents": int(seed["price_cents"]),
            "checkout_url": checkout_url,
        }
    )


def create_missing_catalog_products(limit: int) -> list[dict]:
    created: list[dict] = []
    for _ in range(max(0, limit)):
        item = next_missing_catalog_item()
        if not item:
            break
        created.append(create_product(item))
    return created


def ensure_min_products(min_products: int) -> int:
    existing = count_active_products()
    target = min(len(CATALOG_PRODUCTS), max(1, min_products))
    to_create = max(0, target - existing)
    create_missing_catalog_products(to_create)
    return to_create


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
        ("success_url", f"{base_url}/products/{product['id']}?checkout=success"),
        ("cancel_url", f"{base_url}/products/{product['id']}?checkout=cancel"),
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
            "email_message": "Automated payout from digital store",
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
def landing():
    products = list_products()
    featured = products[:8]
    categories = sorted({p["category"] for p in products})
    return render_template("landing.html", products=featured, categories=categories)


@app.get("/store")
def store():
    products = list_products()
    categories = sorted({p["category"] for p in products})
    return render_template("store.html", products=products, categories=categories)


@app.get("/products/<product_id>")
def product_detail(product_id: str):
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404
    checkout_status = flask_request.args.get("checkout", "")
    return render_template("product.html", product=product, checkout_status=checkout_status)


@app.get("/admin")
@app.get("/dashboard")
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
    return jsonify({"ok": True, "time": utc_now_iso(), "active_products": count_active_products()})


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

    created = create_missing_catalog_products(1)
    if created:
        return jsonify({"product": created[0], "catalog_full": False})

    products = list_products()
    return jsonify({"product": products[0] if products else None, "catalog_full": True})


@app.post("/admin/generate-batch")
def admin_generate_batch():
    if not admin_guard():
        return jsonify({"error": "unauthorized"}), 401

    count = int(flask_request.args.get("count", "8"))
    count = max(1, min(count, 100))
    created = create_missing_catalog_products(count)
    return jsonify({"created_count": len(created), "products": created, "catalog_full": len(created) < count})


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
    interval_seconds = max(30, AUTO_GENERATE_INTERVAL_MINUTES * 60)
    while not stop_event.is_set():
        try:
            create_missing_catalog_products(1)
        except Exception:
            pass
        stop_event.wait(interval_seconds)


def main() -> None:
    init_db()
    deactivate_duplicate_products()
    ensure_min_products(MIN_STORE_PRODUCTS)

    stop_event = threading.Event()
    generator_thread = threading.Thread(target=auto_generator_loop, args=(stop_event,), daemon=True)
    generator_thread.start()

    app.run(host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    main()
