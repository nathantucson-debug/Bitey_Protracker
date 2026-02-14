# Revenue Bot MVP

This is a legal automation starter app for online revenue operations.

It does **not** guarantee instant money. No legal app can reliably guarantee immediate profit. What this app does is automate the pipeline:

- auto-generate digital product listings
- create real Stripe Checkout sessions for product purchases
- accept webhook-confirmed sales (Stripe event endpoint)
- queue and process payouts to Venmo (via PayPal Payouts API when configured)
- provide a dashboard for products and payout status

## Stack

- Python 3.10+
- Flask
- SQLite

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export $(grep -v '^#' .env | xargs)
python app/main.py
```

Open: `http://localhost:8080`

## Go live now (Render)

1. Push this repo to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select your GitHub repo; Render will detect `render.yaml` and deploy.
4. Set required secrets in Render: `APP_PUBLIC_URL`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, `PAYOUT_SENDER_EMAIL`.
5. Open your Render service URL once deployment finishes.

Notes:
- This uses `DATABASE_PATH=/tmp/revenue_bot.db` on Render free tier (ephemeral storage).
- For persistence, switch to Render Postgres or paid disk.

## Key endpoints

- `GET /` dashboard
- `GET /health`
- `GET /api/products`
- `GET /checkout/<product_id>`
- `POST /admin/generate` (header: `x-admin-token`)
- `POST /admin/run-payouts` (header: `x-admin-token`)
- `POST /webhooks/stripe`

## Stripe checkout requirements

Set:

- `STRIPE_SECRET_KEY` (`sk_test_...` in sandbox, `sk_live_...` in production)
- `STRIPE_WEBHOOK_SECRET` from your webhook destination (`whsec_...`)
- `APP_PUBLIC_URL` to your deployed app URL (example: `https://revenue-bot-ktqu.onrender.com`)

Each product checkout link now points to `/checkout/<product_id>` and redirects customers to a Stripe-hosted checkout page.

## Stripe webhook format expected

`POST /webhooks/stripe` expects `checkout.session.completed` payload with:

- `data.object.amount_total`
- `data.object.currency`
- `data.object.metadata.product_id`
- `data.object.metadata.venmo_handle`

And header `Stripe-Signature`, verified against `STRIPE_WEBHOOK_SECRET`.

## Venmo payouts

The app uses PayPal Payouts with `recipient_wallet=VENMO`.

You must configure:

- `PAYPAL_CLIENT_ID`
- `PAYPAL_CLIENT_SECRET`
- `PAYPAL_ENV` (`sandbox` or `live`)
- `PAYOUT_SENDER_EMAIL`

If PayPal credentials are missing, payouts are marked `simulated` so you can test end-to-end behavior safely.

## Production hardening needed

- add authentication and rate limiting
- add retries/dead-letter for payout failures
- add legal pages (terms, privacy, refund policy)
- add analytics and conversion optimization
