# Northstar Studio Storefront

This is a legal automation starter app for online revenue operations.

This app provides a production-style digital storefront pipeline:

- curated product catalog with brand-focused storefront pages
- 50 seeded high-demand digital products across proven categories
- product and bundle checkout with optional order bump
- lead capture for abandoned-cart follow-up
- Stripe checkout + webhook sale confirmation
- post-purchase delivery emails (SMTP or simulated mode)
- payout queue processing to Venmo via PayPal Payouts
- Etsy OAuth connection + draft listing publishing from admin
- Gumroad token-based draft publishing from admin
- Shopify token-based draft publishing from admin

## Stack

- Python 3.10+
- Flask
- SQLite

## Brand + Product Strategy

- Public home page: `/`
- Catalog: `/store`
- Product page: `/products/<product_id>`
- Bundle page: `/bundle/<bundle_key>`
- Market analysis notes: `MARKET_ANALYSIS.md`

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

By default, the app auto-seeds `50` active products (`MIN_STORE_PRODUCTS=50`).

## Go live now (Render)

1. Push this repo to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select your GitHub repo; Render will detect `render.yaml` and deploy.
4. Set required secrets in Render: `APP_PUBLIC_URL`, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET`, `PAYOUT_SENDER_EMAIL`.
5. (Optional) set SMTP vars for automated delivery emails: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `SMTP_FROM`.
6. (Optional) set Etsy vars for marketplace publishing: `ETSY_CLIENT_ID`, `ETSY_CLIENT_SECRET`, `ETSY_REDIRECT_URI`, `ETSY_SCOPES`.
7. (Optional) set Gumroad vars: `GUMROAD_ACCESS_TOKEN`.
8. (Optional) set Shopify vars: `SHOPIFY_STORE_DOMAIN`, `SHOPIFY_ACCESS_TOKEN`, `SHOPIFY_API_VERSION`, `SHOPIFY_CLIENT_ID`, `SHOPIFY_CLIENT_SECRET`, `SHOPIFY_REDIRECT_URI`, `SHOPIFY_SCOPES`.
9. Open your Render service URL once deployment finishes.

Notes:
- This uses `DATABASE_PATH=/tmp/revenue_bot.db` on Render free tier (ephemeral storage).
- For persistence, switch to Render Postgres or paid disk.

## Key endpoints

- `GET /` landing page
- `GET /store` storefront
- `GET /health`
- `GET /api/products`
- `GET /admin/download-links` (admin token required; returns QC ZIP links for all active products)
- `GET /admin/download/product/<product_id>.zip` (admin token required; download product QC package)
- `GET /products/<product_id>`
- `GET /bundle/<bundle_key>`
- `GET /checkout/<product_id>`
- `GET /checkout/bundle/<bundle_key>`
- `POST /capture-lead/product/<product_id>`
- `POST /capture-lead/bundle/<bundle_key>`
- `GET /admin` dashboard
- `GET /connect/etsy` (requires admin token; starts OAuth)
- `GET /connect/etsy/callback` (OAuth callback)
- `POST /admin/publish/etsy/<product_id>` (header `x-admin-token` or form/query `admin_token`)
- `POST /admin/publish/etsy-all` (header `x-admin-token` or form/query `admin_token`)
- `POST /admin/publish/gumroad/<product_id>` (header `x-admin-token` or form/query `admin_token`)
- `POST /admin/publish/gumroad-all` (header `x-admin-token` or form/query `admin_token`)
- `POST /admin/publish/shopify/<product_id>` (header `x-admin-token` or form/query `admin_token`)
- `POST /admin/publish/shopify-all` (header `x-admin-token` or form/query `admin_token`)
- `GET /connect/shopify` (requires admin token; starts Shopify OAuth)
- `GET /connect/shopify/callback` (Shopify OAuth callback)
- `POST /admin/generate` (header: `x-admin-token`)
- `POST /admin/generate-batch?count=10` (header: `x-admin-token`)
- `POST /admin/run-payouts` (header: `x-admin-token`)
- `GET /admin/leads` (header: `x-admin-token`)
- `GET /admin/deliveries` (header: `x-admin-token`)
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
- `data.object.metadata.product_id` or `data.object.metadata.bundle_key`
- `data.object.metadata.lead_id` (optional)
- `data.object.metadata.order_bump` (`yes`/`no`)
- `data.object.customer_details.email` (for delivery email)

And header `Stripe-Signature`, verified against `STRIPE_WEBHOOK_SECRET`.

## Venmo payouts

The app uses PayPal Payouts with `recipient_wallet=VENMO`.

You must configure:

- `PAYPAL_CLIENT_ID`
- `PAYPAL_CLIENT_SECRET`
- `PAYPAL_ENV` (`sandbox` or `live`)
- `PAYOUT_SENDER_EMAIL`

If PayPal credentials are missing, payouts are marked `simulated` so you can test end-to-end behavior safely.

## Etsy integration setup

1. Create an Etsy app in the Etsy developer portal.
2. Add OAuth redirect URI: `https://<your-render-domain>/connect/etsy/callback`.
3. In Render, set:
   - `ETSY_CLIENT_ID`
   - `ETSY_CLIENT_SECRET` (if your Etsy app uses it)
   - `ETSY_REDIRECT_URI` (exactly the callback URL above)
   - `ETSY_SCOPES` (default: `listings_w listings_r shops_r`)
4. Redeploy.
5. Open admin with token in URL for button-based actions:
   - `https://<your-render-domain>/admin?admin_token=<ADMIN_TOKEN>`
6. Click **Connect Etsy** and complete Etsy login/consent.
7. Click **Publish** per product or **Publish All Active Products**.

Notes:
- Published Etsy entries are created as **draft listings** for safe review before public launch.
- This flow uses OAuth; you do not paste Etsy passwords into the app.

## Gumroad integration setup

1. Create a Gumroad API token for your account.
2. In Render set `GUMROAD_ACCESS_TOKEN`.
3. Redeploy and open `https://<your-render-domain>/admin?admin_token=<ADMIN_TOKEN>`.
4. Click **Publish** per product or **Publish All To Gumroad**.

## Shopify integration setup

Option A: OAuth connect (recommended if you already have Dev Dashboard app credentials):
1. In Render set:
   - `SHOPIFY_CLIENT_ID`
   - `SHOPIFY_CLIENT_SECRET`
   - `SHOPIFY_REDIRECT_URI` (example: `https://revenue-bot-ktqu.onrender.com/connect/shopify/callback`)
   - `SHOPIFY_SCOPES` (default `read_products,write_products`)
   - `SHOPIFY_STORE_DOMAIN` (example: `your-store.myshopify.com`)
2. Redeploy.
3. Open:
   - `https://<your-render-domain>/admin?admin_token=<ADMIN_TOKEN>`
4. In Shopify section, click **Connect Shopify** and approve.

Option B: Manual token:
1. In Shopify Admin, create a custom app and grant product write permissions.
2. Install the app and copy the Admin API access token.
3. In Render set:
   - `SHOPIFY_STORE_DOMAIN` (example: `your-store.myshopify.com`)
   - `SHOPIFY_ACCESS_TOKEN`
   - `SHOPIFY_API_VERSION` (default `2024-10`)
4. Redeploy and publish from `/admin`.

## Notes

- Cover assets are in `static/covers/`.
- If SMTP is not configured, delivery emails are logged as simulated.
