import base64
import html
import hashlib
import hmac
import io
import json
import os
import sqlite3
import smtplib
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from email.message import EmailMessage
from urllib import parse, request
from urllib.error import HTTPError, URLError

from flask import Flask, Response, jsonify, redirect, render_template, request as flask_request

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "8080")))
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/revenue_bot.db")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
AUTO_GENERATE_INTERVAL_MINUTES = int(os.getenv("AUTO_GENERATE_INTERVAL_MINUTES", "60"))
MIN_STORE_PRODUCTS = int(os.getenv("MIN_STORE_PRODUCTS", "50"))
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
APP_PUBLIC_URL = os.getenv("APP_PUBLIC_URL", "")
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox")
PAYOUT_SENDER_EMAIL = os.getenv("PAYOUT_SENDER_EMAIL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")
ETSY_CLIENT_ID = os.getenv("ETSY_CLIENT_ID", "")
ETSY_CLIENT_SECRET = os.getenv("ETSY_CLIENT_SECRET", "")
ETSY_REDIRECT_URI = os.getenv("ETSY_REDIRECT_URI", "")
ETSY_SCOPES = os.getenv("ETSY_SCOPES", "listings_w listings_r shops_r")
ETSY_API_BASE = os.getenv("ETSY_API_BASE", "https://api.etsy.com/v3/application")
ETSY_AUTH_BASE = os.getenv("ETSY_AUTH_BASE", "https://www.etsy.com/oauth/connect")
ETSY_TOKEN_URL = os.getenv("ETSY_TOKEN_URL", "https://api.etsy.com/v3/public/oauth/token")
GUMROAD_ACCESS_TOKEN = os.getenv("GUMROAD_ACCESS_TOKEN", "")
GUMROAD_API_BASE = os.getenv("GUMROAD_API_BASE", "https://api.gumroad.com/v2")
SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", "")
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
SHOPIFY_API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2024-10")
SHOPIFY_CLIENT_ID = os.getenv("SHOPIFY_CLIENT_ID", "")
SHOPIFY_CLIENT_SECRET = os.getenv("SHOPIFY_CLIENT_SECRET", "")
SHOPIFY_REDIRECT_URI = os.getenv("SHOPIFY_REDIRECT_URI", "")
SHOPIFY_SCOPES = os.getenv("SHOPIFY_SCOPES", "read_products,write_products")

PAYPAL_BASE = "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"

app = Flask(__name__, template_folder="../templates", static_folder="../static")


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
    {
        "title": "YouTube Script System",
        "category": "Creator Growth",
        "price_cents": 2100,
        "tagline": "Plan faster videos with stronger retention and clearer CTAs.",
        "description": "A reusable script framework for educational and authority-building long-form videos.",
        "preview_items": [
            "8-minute and 12-minute script templates",
            "Retention checkpoints by section",
            "CTA and lead magnet integration prompts",
        ],
        "preview_snippet": "Hook with a painful truth, teach one practical system, then close with a single next step to keep conversions clean.",
    },
    {
        "title": "Instagram Reel Editing Workflow",
        "category": "Creator Growth",
        "price_cents": 1600,
        "tagline": "Edit reels with a repeatable system that saves hours weekly.",
        "description": "A production workflow with shot lists, b-roll map, and cut timing guidance.",
        "preview_items": [
            "Pre-shoot checklist",
            "Editing timeline structure",
            "Caption and thumbnail optimization sheet",
        ],
        "preview_snippet": "Batch your reels by format, not by topic. Speed improves when your editing sequence is fixed.",
    },
    {
        "title": "Newsletter Monetization Toolkit",
        "category": "Creator Business",
        "price_cents": 2400,
        "tagline": "Turn a weekly email list into a dependable revenue channel.",
        "description": "A monetization framework for sponsored placements, offers, and evergreen promos.",
        "preview_items": [
            "Weekly issue layout template",
            "Sponsor placement and pitch scripts",
            "Offer rotation planner",
        ],
        "preview_snippet": "Every issue should have one trust-building section and one conversion section. Consistency beats volume.",
    },
    {
        "title": "Digital Product Launch Emails",
        "category": "Creator Business",
        "price_cents": 1800,
        "tagline": "Launch with persuasive email copy without sounding pushy.",
        "description": "A complete launch sequence from announcement through close-cart follow-up.",
        "preview_items": [
            "7-email launch sequence",
            "Objection handling templates",
            "Post-launch replay emails",
        ],
        "preview_snippet": "Lead with transformation, support with proof, and ask for one clear action per email.",
    },
    {
        "title": "Coaching Program Client Onboarding Kit",
        "category": "Creator Business",
        "price_cents": 2700,
        "tagline": "Create a premium first-week experience for coaching clients.",
        "description": "An onboarding system that reduces confusion and increases client implementation speed.",
        "preview_items": [
            "Welcome packet template",
            "Kickoff session agenda",
            "Progress tracker and accountability check-in forms",
        ],
        "preview_snippet": "Great onboarding sets expectations, goals, and communication rhythm before the first coaching call.",
    },
    {
        "title": "Service Proposal Master Bundle",
        "category": "Freelance Ops",
        "price_cents": 2800,
        "tagline": "Close better clients with polished proposals and scope clarity.",
        "description": "A proposal suite for freelancers and agencies with tiered packages and terms.",
        "preview_items": [
            "Proposal templates by service type",
            "Scope and revision boundaries",
            "Project timeline and payment milestone pages",
        ],
        "preview_snippet": "Specific deliverables and milestones make buyers feel safe and prevent scope drift later.",
    },
    {
        "title": "Client Retainer Renewal Playbook",
        "category": "Freelance Ops",
        "price_cents": 1900,
        "tagline": "Renew contracts with value-focused review calls and data-backed positioning.",
        "description": "A retention framework for turning short engagements into long-term retainers.",
        "preview_items": [
            "Quarterly business review deck",
            "Renewal conversation scripts",
            "Upsell and expansion proposal page",
        ],
        "preview_snippet": "Renewals close faster when you present outcomes, not activity.",
    },
    {
        "title": "Agency KPI Dashboard (Sheets)",
        "category": "Operations",
        "price_cents": 2500,
        "tagline": "Track client, team, and margin metrics from one dashboard.",
        "description": "A KPI command center for small agencies needing weekly decision clarity.",
        "preview_items": [
            "Client profitability tracker",
            "Delivery velocity dashboard",
            "Revenue and utilization summary",
        ],
        "preview_snippet": "Measure margin per client monthly so growth does not hide poor account economics.",
    },
    {
        "title": "Time Blocking Planner Pro",
        "category": "Productivity",
        "price_cents": 1500,
        "tagline": "Design focused workdays with less context switching.",
        "description": "A planner system for deep work, admin batching, and realistic scheduling.",
        "preview_items": [
            "Weekly planning board",
            "Daily block template",
            "Distraction and interruption log",
        ],
        "preview_snippet": "Protect your highest-value work first. Everything else schedules around it.",
    },
    {
        "title": "Executive Assistant Command Center",
        "category": "Productivity",
        "price_cents": 2300,
        "tagline": "Coordinate calendars, follow-ups, and priorities without dropped details.",
        "description": "A digital operations hub for assistants supporting founders and leadership teams.",
        "preview_items": [
            "Priority and escalation board",
            "Meeting prep and follow-up templates",
            "Recurring workflow checklists",
        ],
        "preview_snippet": "Use one source of truth for priorities and owners to avoid inbox-driven execution.",
    },
    {
        "title": "Project Handoff Template Pack",
        "category": "Operations",
        "price_cents": 1700,
        "tagline": "Ship cleaner handoffs between teams and contractors.",
        "description": "A standardized handoff format that cuts confusion and rework.",
        "preview_items": [
            "Project summary template",
            "Dependencies and risks section",
            "Done-definition and QA checklist",
        ],
        "preview_snippet": "Handoffs fail when ownership and next steps are vague. This structure fixes that.",
    },
    {
        "title": "Meeting Notes to Action System",
        "category": "Operations",
        "price_cents": 1400,
        "tagline": "Convert meeting notes into owned, trackable next actions.",
        "description": "A practical system for keeping meetings accountable and execution-oriented.",
        "preview_items": [
            "Action-focused notes template",
            "Owner and due-date tracker",
            "Follow-up summary message scripts",
        ],
        "preview_snippet": "Meetings are valuable only when decisions are documented and assigned immediately.",
    },
    {
        "title": "Debt Snowball Planner + Tracker",
        "category": "Finance",
        "price_cents": 1300,
        "tagline": "Pay down debt with a clear month-by-month execution plan.",
        "description": "A debt reduction workbook with progress visuals and payoff scenarios.",
        "preview_items": [
            "Debt inventory worksheet",
            "Snowball and avalanche comparison view",
            "Monthly payoff tracker",
        ],
        "preview_snippet": "Small wins build momentum. Visual progress helps buyers stay consistent.",
    },
    {
        "title": "Subscription Tracker + Bill Calendar",
        "category": "Finance",
        "price_cents": 1200,
        "tagline": "Stop leaking money to forgotten subscriptions and late fees.",
        "description": "A lightweight finance organizer for recurring bills and renewals.",
        "preview_items": [
            "Subscription inventory tracker",
            "Bill due-date calendar",
            "Annual cost summary",
        ],
        "preview_snippet": "Audit subscriptions quarterly and cancel anything that no longer creates value.",
    },
    {
        "title": "Profit First Allocator Spreadsheet",
        "category": "Finance",
        "price_cents": 2200,
        "tagline": "Allocate revenue into tax, profit, and operating buckets automatically.",
        "description": "A practical Profit First-style spreadsheet for owners who want better cash discipline.",
        "preview_items": [
            "Allocation percentage model",
            "Owner pay projection",
            "Tax reserve tracker",
        ],
        "preview_snippet": "Separate operating cash from profit and tax to reduce financial anxiety during growth.",
    },
    {
        "title": "Pricing Calculator for Freelancers",
        "category": "Freelance Ops",
        "price_cents": 1800,
        "tagline": "Price services for margin, not guesswork.",
        "description": "A calculator that converts workload, costs, and margin goals into profitable prices.",
        "preview_items": [
            "Hourly and project pricing model",
            "Revision-risk buffer field",
            "Target margin simulator",
        ],
        "preview_snippet": "If your calendar is full but cash is tight, pricing is usually the bottleneck.",
    },
    {
        "title": "Job Search CRM (Notion)",
        "category": "Career",
        "price_cents": 1700,
        "tagline": "Run your job search like a pipeline, not a guessing game.",
        "description": "A Notion system for applications, referrals, interviews, and follow-ups.",
        "preview_items": [
            "Application tracker board",
            "Referral outreach log",
            "Interview stage and feedback dashboard",
        ],
        "preview_snippet": "Track every touchpoint so opportunities do not disappear in your inbox.",
    },
    {
        "title": "LinkedIn Authority Content Kit",
        "category": "Career",
        "price_cents": 1900,
        "tagline": "Publish strategic LinkedIn content that attracts opportunities.",
        "description": "A professional content system for career growth and inbound networking.",
        "preview_items": [
            "30 post frameworks",
            "Storytelling and proof templates",
            "Comment-to-conversation CTA prompts",
        ],
        "preview_snippet": "Strong LinkedIn content combines one clear insight with one specific result.",
    },
    {
        "title": "Salary Negotiation Script Pack",
        "category": "Career",
        "price_cents": 1600,
        "tagline": "Negotiate compensation with confidence and structure.",
        "description": "A script and response pack for salary conversations and offer negotiation.",
        "preview_items": [
            "Compensation request scripts",
            "Counteroffer response templates",
            "Total package comparison worksheet",
        ],
        "preview_snippet": "Negotiation works best when you tie your ask to business impact and market benchmarks.",
    },
    {
        "title": "Career Change Roadmap Workbook",
        "category": "Career",
        "price_cents": 2100,
        "tagline": "Plan a strategic transition into a new role or industry.",
        "description": "A step-by-step workbook for skills mapping, portfolio planning, and outreach.",
        "preview_items": [
            "Transferable skills inventory",
            "Target role and company map",
            "90-day transition action plan",
        ],
        "preview_snippet": "A clear plan lowers fear. Focus on one target role and build proof fast.",
    },
    {
        "title": "Wedding Planner Master Spreadsheet",
        "category": "Events",
        "price_cents": 2600,
        "tagline": "Run wedding planning from one clear command center.",
        "description": "A full planning dashboard with budget, vendor, guest, and timeline tools.",
        "preview_items": [
            "Budget and payment tracker",
            "Vendor contact and contract sheet",
            "Timeline and checklist planner",
        ],
        "preview_snippet": "Keep all vendor details in one file to avoid last-minute coordination stress.",
    },
    {
        "title": "Baby Shower Event Kit",
        "category": "Events",
        "price_cents": 1500,
        "tagline": "Host a polished baby shower with less planning friction.",
        "description": "A printable and editable event pack with invites, games, and decor signage.",
        "preview_items": [
            "Invite templates",
            "Game cards and answer sheets",
            "Food and gift tracker",
        ],
        "preview_snippet": "Simple themes and a strong run-of-show make the event feel effortless.",
    },
    {
        "title": "Corporate Event Run-of-Show Template",
        "category": "Events",
        "price_cents": 2900,
        "tagline": "Coordinate speakers, AV, and timing with production-level precision.",
        "description": "A run-of-show framework used by event teams and marketing departments.",
        "preview_items": [
            "Minute-by-minute schedule template",
            "Speaker briefing and cues",
            "Contingency planning checklist",
        ],
        "preview_snippet": "A single source run-of-show prevents production errors and keeps teams aligned.",
    },
    {
        "title": "Restaurant SOP & Training Manual",
        "category": "Hospitality",
        "price_cents": 3400,
        "tagline": "Standardize service quality and onboarding in food businesses.",
        "description": "An SOP and training structure for restaurants and cafe operations.",
        "preview_items": [
            "Opening and closing procedures",
            "New hire training framework",
            "Food safety and quality checklist",
        ],
        "preview_snippet": "Consistency at shift handoff protects guest experience and operational margin.",
    },
    {
        "title": "Vacation Rental Turnover Checklist",
        "category": "Hospitality",
        "price_cents": 1300,
        "tagline": "Improve cleaning speed and quality between guest stays.",
        "description": "A turnover checklist designed for short-term rental owners and cleaning teams.",
        "preview_items": [
            "Room-by-room checklist",
            "Supply restock tracker",
            "Damage report form",
        ],
        "preview_snippet": "Detailed turnovers reduce bad reviews and keep properties five-star ready.",
    },
    {
        "title": "Guest Communication Message Library",
        "category": "Hospitality",
        "price_cents": 1500,
        "tagline": "Respond faster with polished guest messages for every scenario.",
        "description": "A ready-to-use message bank for pre-arrival, support, and checkout communication.",
        "preview_items": [
            "Pre-arrival message sequence",
            "Issue response templates",
            "Review request scripts",
        ],
        "preview_snippet": "Fast, friendly communication improves reviews and reduces avoidable escalations.",
    },
    {
        "title": "Shopify Product Description Swipe File",
        "category": "Ecommerce",
        "price_cents": 1700,
        "tagline": "Write product pages that increase clarity and conversion.",
        "description": "A copywriting library for benefit-driven product descriptions and bullets.",
        "preview_items": [
            "Product page frameworks by category",
            "Benefit and objection template lines",
            "FAQ and comparison blocks",
        ],
        "preview_snippet": "Buyers convert when product pages answer practical questions quickly.",
    },
    {
        "title": "Product Photography Shot List Toolkit",
        "category": "Ecommerce",
        "price_cents": 1600,
        "tagline": "Capture photos that make listings look premium and trustworthy.",
        "description": "A shot planning system for ecommerce brands and handmade product sellers.",
        "preview_items": [
            "Hero and detail shot checklist",
            "Lifestyle and scale shot planner",
            "Image sequence recommendation by platform",
        ],
        "preview_snippet": "Sequence matters. Lead with clarity, then prove quality in detail shots.",
    },
    {
        "title": "Email Welcome Sequence Builder",
        "category": "Ecommerce",
        "price_cents": 2000,
        "tagline": "Turn new subscribers into first-time customers faster.",
        "description": "A welcome flow template for ecommerce and digital product brands.",
        "preview_items": [
            "5-email welcome sequence",
            "Brand story and trust-building prompts",
            "Offer and urgency framework",
        ],
        "preview_snippet": "A welcome sequence should educate, build trust, and ask for one clear purchase action.",
    },
    {
        "title": "Printable Homeschool Planner",
        "category": "Family",
        "price_cents": 1400,
        "tagline": "Organize lessons, routines, and progress in one printable system.",
        "description": "A homeschool planning bundle for daily scheduling and curriculum tracking.",
        "preview_items": [
            "Weekly lesson planner",
            "Subject progress tracker",
            "Attendance and reading log",
        ],
        "preview_snippet": "Clear weekly planning lowers stress and helps families stay consistent.",
    },
    {
        "title": "Family Command Center Calendar",
        "category": "Family",
        "price_cents": 1500,
        "tagline": "Coordinate school, activities, and household routines with less chaos.",
        "description": "A shared planning template for households juggling multiple schedules.",
        "preview_items": [
            "Monthly and weekly family planner",
            "Meal and errand tracker",
            "Responsibilities and reminders board",
        ],
        "preview_snippet": "One visible planning system reduces forgotten tasks and repeated reminders.",
    },
    {
        "title": "Home Cleaning Schedule System",
        "category": "Family",
        "price_cents": 1200,
        "tagline": "Maintain a clean home with a simple recurring routine.",
        "description": "A practical cleaning schedule pack with room zones and recurring checklists.",
        "preview_items": [
            "Daily and weekly task map",
            "Deep-clean rotation schedule",
            "Supply inventory tracker",
        ],
        "preview_snippet": "Short, repeatable routines beat occasional all-day resets.",
    },
    {
        "title": "Healthy Habit Tracker Dashboard",
        "category": "Wellness",
        "price_cents": 1300,
        "tagline": "Build better habits with visual progress and consistency scoring.",
        "description": "A personal wellness tracker for sleep, hydration, movement, and nutrition habits.",
        "preview_items": [
            "Daily habit dashboard",
            "Streak and consistency score",
            "Weekly reflection prompt sheet",
        ],
        "preview_snippet": "Progress is easier when habits are small, visible, and measured weekly.",
    },
    {
        "title": "Workout Program Builder",
        "category": "Wellness",
        "price_cents": 1800,
        "tagline": "Design personalized workout plans clients can follow consistently.",
        "description": "A program design template pack for coaches and fitness creators.",
        "preview_items": [
            "Weekly training split templates",
            "Progression and deload planner",
            "Client check-in template",
        ],
        "preview_snippet": "Simple progression rules improve adherence and reduce program drop-off.",
    },
]

CATALOG_BY_TITLE = {item["title"]: item for item in CATALOG_PRODUCTS}

ORDER_BUMP = {
    "name": "Template Quickstart Video Companion",
    "price_cents": 900,
}

BUNDLES = [
    {
        "key": "creator-revenue",
        "title": "Creator Revenue Bundle",
        "description": "Growth and launch assets for creator-led businesses.",
        "product_titles": ["Creator Caption Vault", "Short-Form Hook Library", "Course Launch Planner"],
        "discount_cents": 1200,
    },
    {
        "key": "freelancer-ops",
        "title": "Freelancer Ops Bundle",
        "description": "Contracts, systems, and finance assets for service operators.",
        "product_titles": ["Freelance Client Pack", "SOP Manual for Small Teams", "Budget & Cashflow Spreadsheet Pro"],
        "discount_cents": 1400,
    },
    {
        "key": "lifestyle-utility",
        "title": "Lifestyle Utility Bundle",
        "description": "Personal productivity and family planning digital systems.",
        "product_titles": ["Meal Prep Planner + Grocery System", "Kids Chore & Reward Chart Pack", "Notion Operator System"],
        "discount_cents": 900,
    },
]

CATEGORY_THEME = {
    "Creator Growth": ("#3b82f6", "#8b5cf6"),
    "Creator Business": ("#0ea5e9", "#2563eb"),
    "Productivity": ("#0891b2", "#0ea5e9"),
    "Finance": ("#059669", "#10b981"),
    "Career": ("#2563eb", "#1d4ed8"),
    "Freelance Ops": ("#7c3aed", "#4f46e5"),
    "Events": ("#db2777", "#f43f5e"),
    "Hospitality": ("#f59e0b", "#d97706"),
    "Ecommerce": ("#0d9488", "#14b8a6"),
    "Wellness": ("#16a34a", "#22c55e"),
    "Family": ("#ea580c", "#f97316"),
    "Operations": ("#334155", "#1f2937"),
    "Branding": ("#4f46e5", "#2563eb"),
    "Real Estate": ("#0369a1", "#0284c7"),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    raw = "".join(c.lower() if c.isalnum() else "-" for c in value).strip("-")
    while "--" in raw:
        raw = raw.replace("--", "-")
    return raw or "product"


def db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def enrich_product(product: dict) -> dict:
    details = CATALOG_BY_TITLE.get(product.get("title", ""), {})
    category = details.get("category", "General")
    theme_start, theme_end = CATEGORY_THEME.get(category, ("#2563eb", "#1d4ed8"))
    product["category"] = category
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
    product["theme_start"] = theme_start
    product["theme_end"] = theme_end
    slug = slugify(product.get("title", "product"))
    static_cover = os.path.join(app.static_folder, "covers", f"{slug}.svg")
    if os.path.exists(static_cover):
        product["cover_image"] = f"/static/covers/{slug}.svg"
    else:
        title_q = parse.quote(product.get("title", "Premium Digital Product"))
        cat_q = parse.quote(category)
        start_q = parse.quote(theme_start)
        end_q = parse.quote(theme_end)
        product["cover_image"] = (
            f"/dynamic-cover.svg?title={title_q}&category={cat_q}&start={start_q}&end={end_q}"
        )
    product["real_world_preview"] = real_world_preview(product.get("title", ""))
    return product


def _wrap_cover_title(text: str, max_chars: int = 28, max_lines: int = 2) -> list[str]:
    words = (text or "Premium Digital Product").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
            if len(lines) >= max_lines - 1:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    if not lines:
        lines = ["Premium Digital Product"]
    return lines[:max_lines]


def build_product_qc_zip(product: dict) -> bytes:
    preview = product.get("real_world_preview", {})
    title = product.get("title", "Untitled Product")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        summary_lines = [
            f"Product: {title}",
            f"Category: {product.get('category', 'General')}",
            f"Price: ${product.get('price_cents', 0) / 100:.2f}",
            "",
            "Tagline:",
            product.get("tagline", ""),
            "",
            "Description:",
            product.get("description", ""),
            "",
            "Included:",
        ]
        for item in product.get("preview_items", []):
            summary_lines.append(f"- {item}")
        summary_lines.extend(
            [
                "",
                "How Buyers Use It:",
                product.get("preview_snippet", ""),
                "",
                "Satisfaction policy:",
                "7-day satisfaction guarantee with one store-credit request per payment.",
            ]
        )
        zf.writestr("README.txt", "\n".join(summary_lines))
        zf.writestr("product.json", json.dumps(product, indent=2))

        artifact_lines = [
            preview.get("headline", "Sample Deliverable"),
            preview.get("subhead", ""),
            "",
            "Columns:",
            ", ".join(preview.get("columns", [])),
            "",
            "Rows:",
        ]
        for row in preview.get("rows", []):
            artifact_lines.append(" | ".join(str(c) for c in row))
        artifact_lines.extend(["", "Outcome:", preview.get("result", "")])
        zf.writestr("sample_artifact.txt", "\n".join(artifact_lines))

        csv_rows = [preview.get("columns", [])] + preview.get("rows", [])
        csv_text = "\n".join(",".join(str(c).replace(",", ";") for c in row) for row in csv_rows if row)
        zf.writestr("sample_artifact.csv", csv_text)

        sales_page = [
            f"# {title}",
            "",
            f"**Category:** {product.get('category', 'General')}",
            f"**Price:** ${product.get('price_cents', 0) / 100:.2f}",
            "",
            f"## {product.get('tagline', '')}",
            "",
            product.get("description", ""),
            "",
            "## Included",
        ]
        for item in product.get("preview_items", []):
            sales_page.append(f"- {item}")
        sales_page.extend(
            [
                "",
                "## Real-World Buyer Outcome",
                preview.get("result", ""),
                "",
                "## Guarantee",
                "7-day satisfaction guarantee. One store-credit request per payment.",
            ]
        )
        zf.writestr("sales_page.md", "\n".join(sales_page))
    return buf.getvalue()


def real_world_preview(title: str) -> dict:
    previews = {
        "Creator Caption Vault": {
            "headline": "Sample Deliverable: 7-Day Instagram Caption Plan",
            "subhead": "A buyer-ready posting schedule with hook, value angle, and CTA.",
            "columns": ["Day", "Hook", "Post Angle", "CTA"],
            "rows": [
                ["Mon", "Nobody tells you this about growth", "Myth vs reality breakdown", "Comment 'map'"],
                ["Tue", "If your content stalls at 300 views", "3 retention fixes", "Save this"],
                ["Wed", "The easiest conversion mistake to fix", "CTA rewrite examples", "DM 'rewrite'"],
                ["Thu", "Steal this post structure", "Hook -> proof -> step", "Share with a creator"],
            ],
            "result": "Outcome: publish 4 high-intent posts with clear conversion goals this week.",
        },
        "Short-Form Hook Library": {
            "headline": "Sample Deliverable: Reels Hook Sheet",
            "subhead": "Real opening lines grouped by awareness level and audience intent.",
            "columns": ["Audience Stage", "Hook Example", "Use Case"],
            "rows": [
                ["Problem aware", "You're not stuck, your first line is.", "Coaching/education"],
                ["Solution aware", "Most creators use this tactic backwards.", "Marketing tips"],
                ["Product aware", "Before you buy another course, fix this first.", "Offer positioning"],
                ["Ready to act", "Copy this script for today's post.", "Template/prompt product"],
            ],
            "result": "Outcome: faster scriptwriting and stronger watch-time from the first 3 seconds.",
        },
        "UGC Pitch Deck Kit": {
            "headline": "Sample Deliverable: UGC Offer Page",
            "subhead": "A client-facing package slide that clarifies scope and pricing.",
            "columns": ["Package", "Deliverables", "Turnaround", "Price"],
            "rows": [
                ["Starter", "3 vertical videos", "5 business days", "$450"],
                ["Growth", "6 vertical videos + hooks", "7 business days", "$850"],
                ["Scale", "10 videos + usage rights", "10 business days", "$1,450"],
            ],
            "result": "Outcome: cleaner proposals and fewer pricing back-and-forth messages.",
        },
        "Notion Operator System": {
            "headline": "Sample Deliverable: Weekly Operator Dashboard",
            "subhead": "A practical snapshot used to run one business week with focus.",
            "columns": ["Priority", "Owner", "Status", "Next Action"],
            "rows": [
                ["Launch lead magnet", "You", "In progress", "Draft checkout page"],
                ["Client onboarding revamp", "You", "Blocked", "Approve form questions"],
                ["Content sprint", "You", "Scheduled", "Record 3 videos Tuesday"],
                ["Finance check-in", "You", "Done", "Export monthly report"],
            ],
            "result": "Outcome: fewer open loops and clearer execution day to day.",
        },
        "Budget & Cashflow Spreadsheet Pro": {
            "headline": "Sample Deliverable: 30-Day Cashflow Snapshot",
            "subhead": "How buyers track income, expenses, and runway in one view.",
            "columns": ["Category", "Planned", "Actual", "Variance"],
            "rows": [
                ["Revenue", "$6,500", "$6,200", "-$300"],
                ["Operating costs", "$1,950", "$1,760", "+$190"],
                ["Owner pay", "$2,200", "$2,200", "$0"],
                ["Net cash movement", "$2,350", "$2,240", "-$110"],
            ],
            "result": "Outcome: immediate visibility into spending leaks and profit decisions.",
        },
        "Resume + Interview Kit": {
            "headline": "Sample Deliverable: Achievement-Focused Resume Section",
            "subhead": "Exactly how a bullet is rewritten for stronger recruiter impact.",
            "columns": ["Before", "After", "Why It Performs Better"],
            "rows": [
                ["Managed campaigns", "Increased qualified leads 34% in 2 quarters", "Quantifies business impact"],
                ["Handled clients", "Owned 22 SMB accounts with 96% retention", "Shows scope + result"],
                ["Improved process", "Cut turnaround time from 5 days to 2 days", "Makes improvement measurable"],
            ],
            "result": "Outcome: stronger interview callbacks with outcome-first positioning.",
        },
        "Freelance Client Pack": {
            "headline": "Sample Deliverable: Scope + Payment Terms Page",
            "subhead": "A real section clients see before they sign.",
            "columns": ["Clause", "Sample Language", "Protection"],
            "rows": [
                ["Project scope", "Includes 3 landing page sections and mobile optimization.", "Limits scope creep"],
                ["Revision policy", "Two revision rounds included within 10 days of delivery.", "Controls rework"],
                ["Payment terms", "50% upfront, 50% before final files are released.", "Protects cashflow"],
            ],
            "result": "Outcome: fewer disputes, clearer expectations, faster payment cycles.",
        },
        "Wedding Invite Suite (Canva)": {
            "headline": "Sample Deliverable: Invitation + RSVP Set",
            "subhead": "A polished wording layout couples personalize in minutes.",
            "columns": ["Card", "Sample Wording", "Format"],
            "rows": [
                ["Main invite", "Together with their families, Ava and Liam invite you...", "5x7 print"],
                ["RSVP", "Kindly reply by September 12. Meal selection included.", "A2 card"],
                ["Details card", "Ceremony 4:00 PM, reception to follow at Lakeside Hall.", "A2 card"],
            ],
            "result": "Outcome: consistent, elegant stationery without custom design fees.",
        },
        "Airbnb Host Welcome Book": {
            "headline": "Sample Deliverable: Guest Arrival Page",
            "subhead": "A practical first page that reduces repetitive host messages.",
            "columns": ["Section", "Sample Content", "Guest Benefit"],
            "rows": [
                ["Wi-Fi", "Network: CasaStay_5G | Password: Welcome2026", "Instant connection"],
                ["Check-out", "11:00 AM. Start dishwasher + lock smart door.", "Clear expectations"],
                ["Local picks", "Best breakfast: Maple House (7 min walk)", "Better guest experience"],
            ],
            "result": "Outcome: fewer support texts and smoother stays.",
        },
        "Etsy Listing SEO Toolkit": {
            "headline": "Sample Deliverable: Etsy Listing Buildout",
            "subhead": "A keyword-focused structure used for one product listing.",
            "columns": ["Element", "Sample Entry", "SEO Purpose"],
            "rows": [
                ["Title", "Editable Wedding Seating Chart Template, Minimalist Canva", "Intent + specificity"],
                ["Primary tags", "wedding seating chart, canva template, printable wedding", "Search discoverability"],
                ["First description line", "Plan your seating in minutes with this editable Canva file.", "Conversion clarity"],
            ],
            "result": "Outcome: improved listing clarity and stronger search alignment.",
        },
        "Meal Prep Planner + Grocery System": {
            "headline": "Sample Deliverable: 5-Day Meal + Grocery Plan",
            "subhead": "A realistic weekly setup that cuts decision fatigue.",
            "columns": ["Day", "Lunch", "Dinner", "Prep Note"],
            "rows": [
                ["Mon", "Chicken rice bowl", "Turkey chili", "Batch cook protein Sunday"],
                ["Tue", "Chicken rice bowl", "Sheet pan salmon", "Use pre-cut vegetables"],
                ["Wed", "Greek wrap", "Turkey chili", "Repurpose leftovers"],
                ["Thu", "Greek wrap", "Veggie pasta", "Prep sauce in advance"],
                ["Fri", "Burrito bowl", "Family pizza night", "Low-prep finish"],
            ],
            "result": "Outcome: 4-6 hours saved weekly and fewer impulse takeout orders.",
        },
        "Kids Chore & Reward Chart Pack": {
            "headline": "Sample Deliverable: Family Chore Tracker (Age 7-10)",
            "subhead": "A real weekly board parents print and place on the fridge.",
            "columns": ["Task", "Mon", "Tue", "Wed", "Thu", "Fri", "Points"],
            "rows": [
                ["Make bed", "✓", "✓", "✓", "✓", "✓", "5"],
                ["Homework done", "✓", "✓", "○", "✓", "✓", "4"],
                ["Feed pet", "✓", "✓", "✓", "○", "✓", "4"],
                ["Evening cleanup", "○", "✓", "✓", "✓", "○", "3"],
            ],
            "result": "Outcome: 16 points earned -> Friday reward selected by the child.",
        },
        "SOP Manual for Small Teams": {
            "headline": "Sample Deliverable: Client Onboarding SOP",
            "subhead": "A true step-by-step process page used by small agencies.",
            "columns": ["Step", "Owner", "Target SLA", "QA Check"],
            "rows": [
                ["Kickoff call scheduled", "Account manager", "24 hours", "Invite + agenda sent"],
                ["Intake form completed", "Client", "48 hours", "Required fields complete"],
                ["Workspace setup", "Operations", "24 hours", "Access + naming rules verified"],
                ["Welcome email sent", "Account manager", "Same day", "Timeline + next steps included"],
            ],
            "result": "Outcome: repeatable onboarding with fewer missed handoffs.",
        },
        "Course Launch Planner": {
            "headline": "Sample Deliverable: Launch Timeline Snapshot",
            "subhead": "A concrete 14-day sequence used before cart open.",
            "columns": ["Day", "Channel", "Asset", "Goal"],
            "rows": [
                ["-14", "Email", "Waitlist warm-up #1", "Re-engage audience"],
                ["-10", "Social", "Problem-awareness reel", "Increase intent"],
                ["-7", "Email", "Case study story", "Build trust"],
                ["-3", "Live", "Q&A session", "Handle objections"],
                ["0", "Email + social", "Cart open announcement", "Drive purchases"],
            ],
            "result": "Outcome: launch messaging stays coordinated instead of last-minute.",
        },
        "Brand Kit + Social Template Bundle": {
            "headline": "Sample Deliverable: Brand Style Snapshot",
            "subhead": "What a buyer gets to ensure visual consistency across channels.",
            "columns": ["Component", "Sample", "Use"],
            "rows": [
                ["Primary color", "Navy #10213D", "Headlines + buttons"],
                ["Secondary color", "Sky #2E8EFF", "Highlights + links"],
                ["Type system", "H1: Sora Bold | Body: Manrope", "Readability + tone"],
                ["Post layout", "Hook / value / CTA structure", "Faster content production"],
            ],
            "result": "Outcome: brand feels cohesive across social, web, and sales pages.",
        },
        "Real Estate Lead Magnet Pack": {
            "headline": "Sample Deliverable: Buyer Follow-Up Sequence",
            "subhead": "A real nurture flow used after a lead downloads your guide.",
            "columns": ["Touchpoint", "Timing", "Message Focus", "CTA"],
            "rows": [
                ["Email #1", "Immediately", "Guide delivery + welcome", "Reply with neighborhood"],
                ["Email #2", "+2 days", "Budget planning checklist", "Book 15-min call"],
                ["Email #3", "+5 days", "Tour readiness tips", "Share target move date"],
                ["Email #4", "+8 days", "Local market update", "Request custom shortlist"],
            ],
            "result": "Outcome: stronger trust and higher conversion from lead to consult.",
        },
    }

    return previews.get(
        title,
        {
            "headline": "Sample Deliverable Preview",
            "subhead": "A practical look at what buyers receive after checkout.",
            "columns": ["Module", "What It Includes", "Buyer Outcome"],
            "rows": [
                ["Quickstart", "Getting started workflow", "Fast activation"],
                ["Core Asset", "Editable template files", "Immediate implementation"],
                ["Optimization", "Checklist + refinements", "Better performance"],
            ],
            "result": "Outcome: day-one usability with premium documentation and support assets.",
        },
    )


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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            source TEXT NOT NULL,
            product_id TEXT,
            bundle_key TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            converted_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS deliveries (
            id TEXT PRIMARY KEY,
            sale_id TEXT NOT NULL,
            email TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS oauth_states (
            state TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            verifier TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS channel_connections (
            provider TEXT PRIMARY KEY,
            access_token TEXT NOT NULL,
            refresh_token TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            account_id TEXT,
            account_name TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS channel_listings (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            product_id TEXT NOT NULL,
            external_id TEXT NOT NULL,
            external_url TEXT,
            status TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
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


def get_bundle(bundle_key: str) -> dict | None:
    for bundle in BUNDLES:
        if bundle["key"] == bundle_key:
            items = [CATALOG_BY_TITLE[t] for t in bundle["product_titles"] if t in CATALOG_BY_TITLE]
            subtotal = sum(int(i["price_cents"]) for i in items)
            discount = int(bundle["discount_cents"])
            price_cents = max(100, subtotal - discount)
            return {
                "key": bundle["key"],
                "title": bundle["title"],
                "description": bundle["description"],
                "items": items,
                "subtotal_cents": subtotal,
                "discount_cents": discount,
                "price_cents": price_cents,
            }
    return None


def list_bundles() -> list[dict]:
    bundles = [b for b in (get_bundle(x["key"]) for x in BUNDLES) if b]
    for bundle in bundles:
        first = bundle["items"][0] if bundle["items"] else None
        if first:
            bundle["cover_image"] = enrich_product({"title": first["title"]})["cover_image"]
            cat = first.get("category", "General")
            s, e = CATEGORY_THEME.get(cat, ("#2563eb", "#1d4ed8"))
            bundle["theme_start"] = s
            bundle["theme_end"] = e
        else:
            bundle["cover_image"] = ""
            bundle["theme_start"] = "#2563eb"
            bundle["theme_end"] = "#1d4ed8"
    return bundles


def is_valid_email(email: str) -> bool:
    if "@" not in email:
        return False
    if "." not in email.split("@", 1)[-1]:
        return False
    return 5 <= len(email) <= 320


def create_lead(email: str, source: str, product_id: str | None = None, bundle_key: str | None = None) -> str:
    lead_id = str(uuid.uuid4())
    conn = db()
    conn.execute(
        "INSERT INTO leads (id, email, source, product_id, bundle_key, status, created_at, converted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (lead_id, email.lower().strip(), source, product_id, bundle_key, "captured", utc_now_iso(), None),
    )
    conn.commit()
    conn.close()
    return lead_id


def mark_lead_converted(lead_id: str) -> None:
    if not lead_id:
        return
    conn = db()
    conn.execute(
        "UPDATE leads SET status = ?, converted_at = ? WHERE id = ?",
        ("converted", utc_now_iso(), lead_id),
    )
    conn.commit()
    conn.close()


def list_recent_abandoned_leads(limit: int = 30) -> list[dict]:
    conn = db()
    rows = conn.execute(
        "SELECT id, email, source, product_id, bundle_key, status, created_at FROM leads WHERE status = 'captured' ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    leads = [dict(r) for r in rows]
    for lead in leads:
        lead["followup_subject"] = "Quick follow-up on your template checkout"
        lead["followup_body"] = (
            "Hi,\\n\\nYou started checkout but did not finish. If you still want the template pack, "
            "you can complete your order here: https://revenue-bot-ktqu.onrender.com/store\\n\\n"
            "Reply if you want help choosing the right product.\\n\\n- Northstar Studio"
        )
    return leads


def list_recent_deliveries(limit: int = 30) -> list[dict]:
    conn = db()
    rows = conn.execute(
        "SELECT id, sale_id, email, status, note, created_at FROM deliveries ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_delivery(sale_id: str, email: str, status: str, note: str) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO deliveries (id, sale_id, email, status, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), sale_id, email, status, note, utc_now_iso()),
    )
    conn.commit()
    conn.close()


def public_base_url() -> str:
    configured = APP_PUBLIC_URL.strip()
    if configured:
        return configured.rstrip("/")
    return flask_request.url_root.rstrip("/")


def admin_token_value() -> str:
    return (
        flask_request.headers.get("x-admin-token")
        or flask_request.args.get("admin_token")
        or flask_request.form.get("admin_token")
        or ""
    )


def admin_guard_any() -> bool:
    return admin_token_value() == ADMIN_TOKEN


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def utc_iso_from_epoch(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def epoch_from_iso(iso_text: str) -> int:
    return int(datetime.fromisoformat(iso_text).timestamp())


def etsy_enabled() -> bool:
    return bool(ETSY_CLIENT_ID and ETSY_REDIRECT_URI)


def gumroad_enabled() -> bool:
    return bool(GUMROAD_ACCESS_TOKEN)


def normalize_shop_domain(value: str) -> str:
    raw = (value or "").strip().lower().replace("https://", "").replace("http://", "").strip("/")
    if raw and not raw.endswith(".myshopify.com"):
        raw = f"{raw}.myshopify.com"
    return raw


def shopify_oauth_enabled() -> bool:
    return bool(SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET and SHOPIFY_REDIRECT_URI)


def shopify_connection() -> dict | None:
    return get_channel_connection("shopify")


def shopify_enabled() -> bool:
    static_ok = bool(normalize_shop_domain(SHOPIFY_STORE_DOMAIN) and SHOPIFY_ACCESS_TOKEN.strip())
    return static_ok or bool(shopify_connection())


def etsy_headers(access_token: str) -> dict[str, str]:
    return {
        "x-api-key": ETSY_CLIENT_ID,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def save_oauth_state(provider: str, state: str, verifier: str) -> None:
    conn = db()
    conn.execute(
        "INSERT INTO oauth_states (state, provider, verifier, created_at) VALUES (?, ?, ?, ?)",
        (state, provider, verifier, utc_now_iso()),
    )
    conn.commit()
    conn.close()


def consume_oauth_state(provider: str, state: str) -> str | None:
    conn = db()
    row = conn.execute(
        "SELECT verifier FROM oauth_states WHERE state = ? AND provider = ?",
        (state, provider),
    ).fetchone()
    conn.execute("DELETE FROM oauth_states WHERE state = ? AND provider = ?", (state, provider))
    conn.commit()
    conn.close()
    return row["verifier"] if row else None


def save_channel_connection(
    *,
    provider: str,
    access_token: str,
    refresh_token: str,
    expires_at: str,
    account_id: str = "",
    account_name: str = "",
) -> None:
    conn = db()
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO channel_connections (provider, access_token, refresh_token, expires_at, account_id, account_name, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider) DO UPDATE SET
            access_token = excluded.access_token,
            refresh_token = excluded.refresh_token,
            expires_at = excluded.expires_at,
            account_id = excluded.account_id,
            account_name = excluded.account_name,
            updated_at = excluded.updated_at
        """,
        (provider, access_token, refresh_token, expires_at, account_id, account_name, now, now),
    )
    conn.commit()
    conn.close()


def get_channel_connection(provider: str) -> dict | None:
    conn = db()
    row = conn.execute(
        "SELECT provider, access_token, refresh_token, expires_at, account_id, account_name, created_at, updated_at FROM channel_connections WHERE provider = ?",
        (provider,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_channel_listings(provider: str, limit: int = 100) -> list[dict]:
    conn = db()
    rows = conn.execute(
        "SELECT id, provider, product_id, external_id, external_url, status, note, created_at, updated_at FROM channel_listings WHERE provider = ? ORDER BY updated_at DESC LIMIT ?",
        (provider, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def upsert_channel_listing(
    *,
    provider: str,
    product_id: str,
    external_id: str,
    external_url: str,
    status: str,
    note: str,
) -> None:
    conn = db()
    now = utc_now_iso()
    existing = conn.execute(
        "SELECT id FROM channel_listings WHERE provider = ? AND product_id = ?",
        (provider, product_id),
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE channel_listings
            SET external_id = ?, external_url = ?, status = ?, note = ?, updated_at = ?
            WHERE id = ?
            """,
            (external_id, external_url, status, note, now, existing["id"]),
        )
    else:
        conn.execute(
            """
            INSERT INTO channel_listings (id, provider, product_id, external_id, external_url, status, note, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), provider, product_id, external_id, external_url, status, note, now, now),
        )
    conn.commit()
    conn.close()


def etsy_token_request(payload: dict) -> dict:
    req = request.Request(
        ETSY_TOKEN_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def etsy_auth_url(state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "redirect_uri": ETSY_REDIRECT_URI,
        "scope": ETSY_SCOPES,
        "client_id": ETSY_CLIENT_ID,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{ETSY_AUTH_BASE}?{parse.urlencode(params)}"


def etsy_exchange_code(code: str, verifier: str) -> dict:
    payload = {
        "grant_type": "authorization_code",
        "client_id": ETSY_CLIENT_ID,
        "redirect_uri": ETSY_REDIRECT_URI,
        "code": code,
        "code_verifier": verifier,
    }
    if ETSY_CLIENT_SECRET:
        payload["client_secret"] = ETSY_CLIENT_SECRET
    return etsy_token_request(payload)


def etsy_refresh_access_token(refresh_token: str) -> dict:
    payload = {
        "grant_type": "refresh_token",
        "client_id": ETSY_CLIENT_ID,
        "refresh_token": refresh_token,
    }
    if ETSY_CLIENT_SECRET:
        payload["client_secret"] = ETSY_CLIENT_SECRET
    return etsy_token_request(payload)


def etsy_get_json(path: str, access_token: str) -> dict:
    req = request.Request(
        f"{ETSY_API_BASE}{path}",
        headers=etsy_headers(access_token),
        method="GET",
    )
    with request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def etsy_post_json(path: str, access_token: str, payload: dict) -> dict:
    req = request.Request(
        f"{ETSY_API_BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=etsy_headers(access_token),
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def etsy_detect_shop(access_token: str) -> tuple[str, str]:
    user = etsy_get_json("/users/me", access_token)
    user_id = str(user.get("user_id") or user.get("userId") or "")
    if user_id:
        shops = etsy_get_json(f"/users/{user_id}/shops", access_token)
        results = shops.get("results") or []
        if results:
            shop = results[0]
            return str(shop.get("shop_id") or shop.get("shopId") or ""), str(shop.get("shop_name") or shop.get("shopName") or "")

    shops = etsy_get_json("/shops?limit=1", access_token)
    results = shops.get("results") or []
    if results:
        shop = results[0]
        return str(shop.get("shop_id") or shop.get("shopId") or ""), str(shop.get("shop_name") or shop.get("shopName") or "")
    return "", ""


def get_valid_etsy_connection() -> dict:
    conn_data = get_channel_connection("etsy")
    if not conn_data:
        raise ValueError("etsy not connected")

    expires_at = conn_data.get("expires_at", "")
    refresh_token = conn_data.get("refresh_token", "")
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    refresh_needed = True
    if expires_at:
        try:
            refresh_needed = epoch_from_iso(expires_at) <= (now_epoch + 120)
        except Exception:
            refresh_needed = True

    if refresh_needed:
        refreshed = etsy_refresh_access_token(refresh_token)
        expires_in = int(refreshed.get("expires_in") or 3600)
        expires_iso = utc_iso_from_epoch(now_epoch + expires_in)
        access_token = refreshed.get("access_token", conn_data["access_token"])
        new_refresh = refreshed.get("refresh_token", refresh_token)
        account_id = conn_data.get("account_id", "")
        account_name = conn_data.get("account_name", "")
        save_channel_connection(
            provider="etsy",
            access_token=access_token,
            refresh_token=new_refresh,
            expires_at=expires_iso,
            account_id=account_id,
            account_name=account_name,
        )
        conn_data = get_channel_connection("etsy") or conn_data
    return conn_data


def etsy_listing_payload(product: dict) -> dict:
    tags = [t.lower().replace(" ", "")[:20] for t in product.get("category", "digital").split("/") if t]
    title_words = [w for w in product["title"].split(" ") if w]
    for word in title_words[:10]:
        tag = "".join(c for c in word.lower() if c.isalnum())[:20]
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) >= 13:
            break
    return {
        "title": product["title"][:140],
        "description": f"{product['description']}\n\nIncludes:\n- " + "\n- ".join(product.get("preview_items", [])),
        "price": round(int(product["price_cents"]) / 100, 2),
        "quantity": 999,
        "who_made": "i_did",
        "when_made": "made_to_order",
        "is_supply": False,
        "state": "draft",
        "type": "download",
        "tags": tags[:13],
    }


def publish_product_to_etsy(product: dict) -> dict:
    conn_data = get_valid_etsy_connection()
    shop_id = conn_data.get("account_id", "")
    access_token = conn_data.get("access_token", "")
    if not shop_id:
        raise ValueError("etsy shop not found; reconnect Etsy")

    payload = etsy_listing_payload(product)
    response = etsy_post_json(f"/shops/{shop_id}/listings", access_token, payload)
    listing_id = str(response.get("listing_id") or response.get("listingId") or "")
    if not listing_id:
        raise ValueError(f"etsy listing create failed: {response}")

    listing_url = f"https://www.etsy.com/listing/{listing_id}"
    upsert_channel_listing(
        provider="etsy",
        product_id=product["id"],
        external_id=listing_id,
        external_url=listing_url,
        status="draft",
        note="Created as Etsy draft listing",
    )
    return {"listing_id": listing_id, "listing_url": listing_url, "status": "draft"}


def publish_product_to_gumroad(product: dict) -> dict:
    if not gumroad_enabled():
        raise ValueError("gumroad not configured")

    payload = {
        "access_token": GUMROAD_ACCESS_TOKEN,
        "name": product["title"],
        "price": int(product["price_cents"]),
        "description": f"{product['description']}\n\nIncludes:\n- " + "\n- ".join(product.get("preview_items", [])),
        "published": False,
    }
    req = request.Request(
        f"{GUMROAD_API_BASE}/products",
        data=parse.urlencode(payload).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    product_obj = body.get("product") or body.get("resource") or body
    external_id = str(product_obj.get("id") or "")
    if not external_id:
        raise ValueError(f"gumroad create failed: {body}")
    external_url = str(product_obj.get("short_url") or product_obj.get("url") or "")
    upsert_channel_listing(
        provider="gumroad",
        product_id=product["id"],
        external_id=external_id,
        external_url=external_url,
        status="draft",
        note="Created as Gumroad draft product",
    )
    return {"product_id": external_id, "product_url": external_url, "status": "draft"}


def verify_shopify_hmac(args_dict, shared_secret: str) -> bool:
    given = (args_dict.get("hmac") or "").strip()
    if not given:
        return False
    pairs = []
    for key in sorted(args_dict.keys()):
        if key in {"hmac", "signature"}:
            continue
        values = args_dict.getlist(key)
        if not values:
            pairs.append(f"{key}=")
            continue
        for value in sorted(values):
            pairs.append(f"{key}={value}")
    message = "&".join(pairs)
    digest = hmac.new(shared_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, given)


def make_signed_oauth_state(provider: str, payload: dict, secret: str) -> str:
    body = {"provider": provider, "ts": int(datetime.now(timezone.utc).timestamp()), **payload}
    payload_b64 = b64url(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"


def parse_signed_oauth_state(state: str, secret: str, max_age_seconds: int = 900) -> dict | None:
    parts = (state or "").split(".", 1)
    if len(parts) != 2:
        return None
    payload_b64, provided_sig = parts
    expected_sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, provided_sig):
        return None
    try:
        body = json.loads(b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None
    ts = int(body.get("ts") or 0)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if ts <= 0 or (now_ts - ts) > max_age_seconds:
        return None
    return body


def exchange_shopify_oauth_code(shop: str, code: str) -> dict:
    endpoint = f"https://{shop}/admin/oauth/access_token"
    payload = {
        "client_id": SHOPIFY_CLIENT_ID,
        "client_secret": SHOPIFY_CLIENT_SECRET,
        "code": code,
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_shopify_credentials() -> tuple[str, str]:
    static_domain = normalize_shop_domain(SHOPIFY_STORE_DOMAIN)
    static_token = SHOPIFY_ACCESS_TOKEN.strip()
    if static_domain and static_token:
        return static_domain, static_token

    conn_data = shopify_connection()
    if conn_data:
        domain = normalize_shop_domain(conn_data.get("account_id", ""))
        token = (conn_data.get("access_token") or "").strip()
        if domain and token:
            return domain, token

    raise ValueError("shopify not connected")


def publish_product_to_shopify(product: dict) -> dict:
    if not shopify_enabled():
        raise ValueError("shopify not configured")

    domain, access_token = resolve_shopify_credentials()
    endpoint = f"https://{domain}/admin/api/{SHOPIFY_API_VERSION}/products.json"
    payload = {
        "product": {
            "title": product["title"],
            "body_html": "<p>"
            + product["description"]
            + "</p><ul>"
            + "".join(f"<li>{item}</li>" for item in product.get("preview_items", []))
            + "</ul>",
            "vendor": "Northstar Studio",
            "product_type": product.get("category", "Digital Product"),
            "status": "draft",
            "variants": [{"price": f"{int(product['price_cents']) / 100:.2f}"}],
        }
    }
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "X-Shopify-Access-Token": access_token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    product_obj = body.get("product") or {}
    external_id = str(product_obj.get("id") or "")
    if not external_id:
        raise ValueError(f"shopify create failed: {body}")
    handle = product_obj.get("handle") or ""
    external_url = f"https://{domain}/products/{handle}" if handle else ""
    upsert_channel_listing(
        provider="shopify",
        product_id=product["id"],
        external_id=external_id,
        external_url=external_url,
        status="draft",
        note="Created as Shopify draft product",
    )
    return {"product_id": external_id, "product_url": external_url, "status": "draft"}


def create_stripe_checkout_session(
    *,
    line_items: list[dict],
    success_url: str,
    cancel_url: str,
    metadata: dict[str, str],
) -> str:
    if not STRIPE_SECRET_KEY:
        raise ValueError("missing STRIPE_SECRET_KEY")

    payload: list[tuple[str, str]] = [("mode", "payment"), ("success_url", success_url), ("cancel_url", cancel_url)]
    for idx, item in enumerate(line_items):
        payload.extend(
            [
                (f"line_items[{idx}][price_data][currency]", "usd"),
                (f"line_items[{idx}][price_data][unit_amount]", str(int(item["price_cents"]))),
                (f"line_items[{idx}][price_data][product_data][name]", item["name"]),
                (f"line_items[{idx}][quantity]", "1"),
            ]
        )

    for k, v in metadata.items():
        if v:
            payload.append((f"metadata[{k}]", v))

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


def create_product_checkout_session(product: dict, add_bump: bool, lead_id: str) -> str:
    base_url = public_base_url()
    items = [{"name": product["title"], "price_cents": int(product["price_cents"])}]
    if add_bump:
        items.append({"name": ORDER_BUMP["name"], "price_cents": int(ORDER_BUMP["price_cents"])})

    return create_stripe_checkout_session(
        line_items=items,
        success_url=f"{base_url}/products/{product['id']}?checkout=success",
        cancel_url=f"{base_url}/products/{product['id']}?checkout=cancel",
        metadata={
            "product_id": product["id"],
            "lead_id": lead_id,
            "order_bump": "yes" if add_bump else "no",
            "venmo_handle": PAYOUT_SENDER_EMAIL,
        },
    )


def create_bundle_checkout_session(bundle: dict, add_bump: bool, lead_id: str) -> str:
    base_url = public_base_url()
    line_items = [{"name": i["title"], "price_cents": int(i["price_cents"])} for i in bundle["items"]]
    # Apply bundle discount as a single negative-cost equivalent by reducing each line item's price proportionally.
    # Simpler for MVP: replace with one bundle line item at discounted total.
    line_items = [{"name": bundle["title"], "price_cents": int(bundle["price_cents"])}]
    if add_bump:
        line_items.append({"name": ORDER_BUMP["name"], "price_cents": int(ORDER_BUMP["price_cents"])})

    return create_stripe_checkout_session(
        line_items=line_items,
        success_url=f"{base_url}/bundle/{bundle['key']}?checkout=success",
        cancel_url=f"{base_url}/bundle/{bundle['key']}?checkout=cancel",
        metadata={
            "bundle_key": bundle["key"],
            "lead_id": lead_id,
            "order_bump": "yes" if add_bump else "no",
            "venmo_handle": PAYOUT_SENDER_EMAIL,
        },
    )


def send_email(to_email: str, subject: str, body: str) -> tuple[bool, str]:
    if not SMTP_HOST or not SMTP_FROM:
        return True, "simulated (smtp not configured)"
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            if SMTP_USER:
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True, "sent"
    except Exception as exc:
        return False, str(exc)


def handle_post_purchase_delivery(event: dict, sale_id: str) -> tuple[bool, str]:
    data_object = event.get("data", {}).get("object", {})
    metadata = data_object.get("metadata", {})
    customer_details = data_object.get("customer_details", {}) or {}
    email = (customer_details.get("email") or data_object.get("customer_email") or "").strip().lower()
    if not email:
        return False, "missing customer email in checkout session"

    product_id = metadata.get("product_id", "")
    bundle_key = metadata.get("bundle_key", "")
    order_bump = metadata.get("order_bump", "no")
    base_url = APP_PUBLIC_URL.strip() or "https://revenue-bot-ktqu.onrender.com"

    lines = [
        "Thanks for your purchase from Northstar Studio.",
        "",
        "Your access links:",
    ]
    if product_id:
        product = get_product(product_id)
        if product:
            lines.append(f"- {product['title']}: {base_url}/products/{product_id}")
    if bundle_key:
        lines.append(f"- Bundle access page: {base_url}/bundle/{bundle_key}")
    if order_bump == "yes":
        lines.append(f"- {ORDER_BUMP['name']}: {base_url}/quickstart-video-companion")
    lines.extend(
        [
            "",
            "Support policy: 7-day satisfaction guarantee with one store-credit request per payment.",
            "Store credit is issued for the full purchase amount and can be redeemed toward another product.",
            "To request it, reply to this email within 7 days of purchase.",
            "",
            "Northstar Studio",
        ]
    )
    ok, note = send_email(email, "Your Northstar Studio purchase access", "\n".join(lines))
    log_delivery(sale_id=sale_id, email=email, status="sent" if ok else "failed", note=note)
    return ok, note


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
    venmo_handle = data_object.get("metadata", {}).get("venmo_handle") or PAYOUT_SENDER_EMAIL or ""
    lead_id = data_object.get("metadata", {}).get("lead_id") or ""
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
    mark_lead_converted(lead_id)
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
    return render_template("landing.html", products=featured, categories=categories, bundles=list_bundles())


@app.get("/dynamic-cover.svg")
def dynamic_cover() -> Response:
    title = flask_request.args.get("title", "Premium Digital Product")
    category = flask_request.args.get("category", "Northstar Studio")
    start = flask_request.args.get("start", "#2563eb")
    end = flask_request.args.get("end", "#1d4ed8")
    lines = _wrap_cover_title(title)

    safe_category = html.escape(category[:40])
    safe_lines = [html.escape(line[:40]) for line in lines]

    line_y_start = 278
    line_gap = 78
    line_markup = "".join(
        f'<text x="84" y="{line_y_start + idx * line_gap}" fill="#ffffff" '
        f'font-size="58" font-family="Sora, Arial, sans-serif" font-weight="800">{line}</text>'
        for idx, line in enumerate(safe_lines)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="675" viewBox="0 0 1200 675">
<defs>
  <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="{html.escape(start)}"/>
    <stop offset="100%" stop-color="{html.escape(end)}"/>
  </linearGradient>
  <linearGradient id="glass" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="rgba(255,255,255,0.28)"/>
    <stop offset="100%" stop-color="rgba(255,255,255,0.08)"/>
  </linearGradient>
</defs>
<rect width="1200" height="675" rx="28" fill="url(#g)"/>
<rect x="54" y="54" width="1092" height="567" rx="24" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="2"/>
<circle cx="1044" cy="114" r="78" fill="rgba(255,255,255,0.09)"/>
<circle cx="1005" cy="540" r="122" fill="rgba(255,255,255,0.07)"/>
<rect x="84" y="72" width="290" height="56" rx="28" fill="rgba(255,255,255,0.18)"/>
<text x="116" y="109" fill="#ffffff" font-size="34" font-family="Sora, Arial, sans-serif" font-weight="700">Northstar Studio</text>
<text x="84" y="178" fill="rgba(255,255,255,0.92)" font-size="38" font-family="Manrope, Arial, sans-serif">{safe_category}</text>
{line_markup}
<rect x="84" y="472" width="1032" height="2" fill="rgba(255,255,255,0.3)"/>
<rect x="84" y="500" width="260" height="122" rx="14" fill="url(#glass)" stroke="rgba(255,255,255,0.25)" />
<text x="106" y="542" fill="#ffffff" font-size="27" font-family="Sora, Arial, sans-serif" font-weight="700">Instant Download</text>
<text x="106" y="575" fill="rgba(255,255,255,0.92)" font-size="23" font-family="Manrope, Arial, sans-serif">Editable files included</text>
<text x="84" y="560" fill="rgba(255,255,255,0.92)" font-size="30" font-family="Manrope, Arial, sans-serif">Premium Digital Product</text>
</svg>"""
    return Response(svg, mimetype="image/svg+xml")


@app.get("/store")
def store():
    products = list_products()
    categories = sorted({p["category"] for p in products})
    selected_category = (flask_request.args.get("category") or "").strip()
    if selected_category:
        products = [p for p in products if p["category"] == selected_category]
    return render_template(
        "store.html",
        products=products,
        categories=categories,
        bundles=list_bundles(),
        selected_category=selected_category,
    )


@app.get("/products/<product_id>")
def product_detail(product_id: str):
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404
    checkout_status = flask_request.args.get("checkout", "")
    lead_id = flask_request.args.get("lead_id", "")
    lead_status = flask_request.args.get("lead", "")
    return render_template(
        "product.html",
        product=product,
        checkout_status=checkout_status,
        lead_id=lead_id,
        lead_status=lead_status,
        order_bump=ORDER_BUMP,
    )


@app.get("/bundle/<bundle_key>")
def bundle_detail(bundle_key: str):
    bundle = get_bundle(bundle_key)
    if not bundle:
        return jsonify({"error": "bundle not found"}), 404
    checkout_status = flask_request.args.get("checkout", "")
    lead_id = flask_request.args.get("lead_id", "")
    lead_status = flask_request.args.get("lead", "")
    return render_template(
        "bundle.html",
        bundle=bundle,
        checkout_status=checkout_status,
        lead_id=lead_id,
        lead_status=lead_status,
        order_bump=ORDER_BUMP,
    )


@app.get("/quickstart-video-companion")
def quickstart_video_companion():
    return render_template("quickstart_video_companion.html", order_bump=ORDER_BUMP)


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
    leads = list_recent_abandoned_leads(limit=20)
    deliveries = list_recent_deliveries(limit=20)
    etsy_conn = get_channel_connection("etsy")
    shopify_conn = shopify_connection()
    etsy_listings = list_channel_listings("etsy", limit=50)
    gumroad_listings = list_channel_listings("gumroad", limit=50)
    shopify_listings = list_channel_listings("shopify", limit=50)
    return render_template(
        "index.html",
        products=products,
        payouts=payouts,
        leads=leads,
        deliveries=deliveries,
        etsy_connected=bool(etsy_conn),
        etsy_account_name=(etsy_conn or {}).get("account_name", ""),
        etsy_shop_id=(etsy_conn or {}).get("account_id", ""),
        etsy_listings=etsy_listings,
        gumroad_listings=gumroad_listings,
        shopify_listings=shopify_listings,
        admin_token_hint=(flask_request.args.get("admin_token") or ""),
        etsy_enabled=etsy_enabled(),
        gumroad_enabled=gumroad_enabled(),
        shopify_enabled=shopify_enabled(),
        shopify_oauth_enabled=shopify_oauth_enabled(),
        shopify_connected=bool(shopify_conn) or bool(normalize_shop_domain(SHOPIFY_STORE_DOMAIN) and SHOPIFY_ACCESS_TOKEN.strip()),
        shopify_store=(shopify_conn or {}).get("account_id", normalize_shop_domain(SHOPIFY_STORE_DOMAIN)),
    )


@app.get("/health")
def health():
    return jsonify({"ok": True, "time": utc_now_iso(), "active_products": count_active_products()})


@app.get("/api/products")
def api_products():
    return jsonify({"products": list_products()})


@app.get("/admin/download/product/<product_id>.zip")
def admin_download_product_zip(product_id: str):
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404
    payload = build_product_qc_zip(product)
    filename = f"{slugify(product.get('title', 'product'))}-qc-pack.zip"
    return Response(
        payload,
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/admin/download-links")
def admin_download_links():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    token = admin_token_value()
    base = public_base_url()
    links = []
    for p in list_products():
        links.append(
            {
                "id": p["id"],
                "title": p["title"],
                "qc_download": f"{base}/admin/download/product/{p['id']}.zip?admin_token={parse.quote(token)}",
                "preview": f"{base}/products/{p['id']}",
                "checkout": f"{base}/checkout/{p['id']}",
            }
        )
    return jsonify({"count": len(links), "products": links})


@app.get("/checkout/<product_id>")
def checkout(product_id: str):
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404

    lead_id = (flask_request.args.get("lead_id") or "").strip()
    add_bump = flask_request.args.get("add_bump", "0") == "1"
    try:
        session_url = create_product_checkout_session(product, add_bump=add_bump, lead_id=lead_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": "stripe api error", "detail": detail}), 502
    except URLError as exc:
        return jsonify({"error": "stripe network error", "detail": str(exc)}), 502

    return redirect(session_url, code=302)


@app.get("/checkout/bundle/<bundle_key>")
def checkout_bundle(bundle_key: str):
    bundle = get_bundle(bundle_key)
    if not bundle:
        return jsonify({"error": "bundle not found"}), 404

    lead_id = (flask_request.args.get("lead_id") or "").strip()
    add_bump = flask_request.args.get("add_bump", "0") == "1"
    try:
        session_url = create_bundle_checkout_session(bundle, add_bump=add_bump, lead_id=lead_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": "stripe api error", "detail": detail}), 502
    except URLError as exc:
        return jsonify({"error": "stripe network error", "detail": str(exc)}), 502

    return redirect(session_url, code=302)


@app.post("/capture-lead/product/<product_id>")
def capture_product_lead(product_id: str):
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404

    email = (flask_request.form.get("email") or "").strip().lower()
    if not is_valid_email(email):
        return redirect(f"/products/{product_id}?lead=invalid")

    lead_id = create_lead(email=email, source="product_page", product_id=product_id)
    return redirect(f"/products/{product_id}?lead=captured&lead_id={lead_id}")


@app.post("/capture-lead/bundle/<bundle_key>")
def capture_bundle_lead(bundle_key: str):
    bundle = get_bundle(bundle_key)
    if not bundle:
        return jsonify({"error": "bundle not found"}), 404

    email = (flask_request.form.get("email") or "").strip().lower()
    if not is_valid_email(email):
        return redirect(f"/bundle/{bundle_key}?lead=invalid")

    lead_id = create_lead(email=email, source="bundle_page", bundle_key=bundle_key)
    return redirect(f"/bundle/{bundle_key}?lead=captured&lead_id={lead_id}")


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


@app.get("/connect/etsy")
def connect_etsy():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    if not etsy_enabled():
        return jsonify({"error": "etsy not configured", "needed": ["ETSY_CLIENT_ID", "ETSY_REDIRECT_URI"]}), 503

    verifier = b64url(os.urandom(32))
    challenge = b64url(hashlib.sha256(verifier.encode("utf-8")).digest())
    state = str(uuid.uuid4())
    save_oauth_state("etsy", state, verifier)
    return redirect(etsy_auth_url(state=state, code_challenge=challenge), code=302)


@app.get("/connect/etsy/callback")
def connect_etsy_callback():
    state = (flask_request.args.get("state") or "").strip()
    code = (flask_request.args.get("code") or "").strip()
    error = (flask_request.args.get("error") or "").strip()
    if error:
        return jsonify({"error": "etsy oauth failed", "detail": error}), 400
    if not state or not code:
        return jsonify({"error": "missing oauth code/state"}), 400

    verifier = consume_oauth_state("etsy", state)
    if not verifier:
        return jsonify({"error": "invalid oauth state"}), 400

    try:
        token_data = etsy_exchange_code(code=code, verifier=verifier)
        access_token = token_data.get("access_token", "")
        refresh_token = token_data.get("refresh_token", "")
        expires_in = int(token_data.get("expires_in") or 3600)
        expires_at = utc_iso_from_epoch(int(datetime.now(timezone.utc).timestamp()) + expires_in)
        if not access_token or not refresh_token:
            return jsonify({"error": "etsy token exchange failed", "detail": token_data}), 502
        shop_id, shop_name = etsy_detect_shop(access_token)
        save_channel_connection(
            provider="etsy",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            account_id=shop_id,
            account_name=shop_name,
        )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": "etsy api error", "detail": detail}), 502
    except URLError as exc:
        return jsonify({"error": "etsy network error", "detail": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": "etsy connect failed", "detail": str(exc)}), 500

    return redirect("/admin?etsy=connected", code=302)


@app.post("/admin/publish/etsy/<product_id>")
def admin_publish_etsy(product_id: str):
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401

    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404

    try:
        listing = publish_product_to_etsy(product)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": "etsy api error", "detail": detail}), 502
    except URLError as exc:
        return jsonify({"error": "etsy network error", "detail": str(exc)}), 502

    if flask_request.form.get("redirect") == "1":
        return redirect("/admin?etsy=published", code=302)
    return jsonify({"ok": True, "product_id": product_id, "listing": listing})


@app.post("/admin/publish/etsy-all")
def admin_publish_etsy_all():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401

    products = list_products(active_only=True)
    results = []
    for product in products:
        try:
            listing = publish_product_to_etsy(product)
            results.append({"product_id": product["id"], "title": product["title"], "ok": True, "listing": listing})
        except Exception as exc:
            results.append({"product_id": product["id"], "title": product["title"], "ok": False, "error": str(exc)})

    success_count = len([r for r in results if r["ok"]])
    fail_count = len(results) - success_count
    if flask_request.form.get("redirect") == "1":
        return redirect("/admin?etsy=bulk", code=302)
    return jsonify({"ok": True, "success": success_count, "failed": fail_count, "results": results})


@app.get("/connect/shopify")
def connect_shopify():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    if not shopify_oauth_enabled():
        return (
            jsonify(
                {
                    "error": "shopify oauth not configured",
                    "needed": ["SHOPIFY_CLIENT_ID", "SHOPIFY_CLIENT_SECRET", "SHOPIFY_REDIRECT_URI"],
                }
            ),
            503,
        )

    shop = normalize_shop_domain(flask_request.args.get("shop") or SHOPIFY_STORE_DOMAIN)
    if not shop:
        return jsonify({"error": "missing shop parameter", "example": "/connect/shopify?shop=your-store.myshopify.com"}), 400

    state = make_signed_oauth_state("shopify", {"shop": shop}, SHOPIFY_CLIENT_SECRET)
    params = {
        "client_id": SHOPIFY_CLIENT_ID,
        "scope": SHOPIFY_SCOPES,
        "redirect_uri": SHOPIFY_REDIRECT_URI,
        "state": state,
    }
    return redirect(f"https://{shop}/admin/oauth/authorize?{parse.urlencode(params)}", code=302)


@app.get("/connect/shopify/callback")
def connect_shopify_callback():
    if not shopify_oauth_enabled():
        return jsonify({"error": "shopify oauth not configured"}), 503

    shop = normalize_shop_domain(flask_request.args.get("shop", ""))
    code = (flask_request.args.get("code") or "").strip()
    state = (flask_request.args.get("state") or "").strip()
    if not shop or not code:
        return jsonify({"error": "missing oauth fields"}), 400
    if not verify_shopify_hmac(flask_request.args, SHOPIFY_CLIENT_SECRET):
        return jsonify({"error": "invalid shopify hmac"}), 400
    # Accept callback if Shopify HMAC is valid and shop matches configured store.
    configured_shop = normalize_shop_domain(SHOPIFY_STORE_DOMAIN)
    if configured_shop and configured_shop != shop:
        return jsonify({"error": "shop mismatch"}), 400

    try:
        token_data = exchange_shopify_oauth_code(shop=shop, code=code)
        access_token = (token_data.get("access_token") or "").strip()
        if not access_token:
            return jsonify({"error": "shopify token exchange failed", "detail": token_data}), 502
        save_channel_connection(
            provider="shopify",
            access_token=access_token,
            refresh_token="",
            expires_at="2099-01-01T00:00:00+00:00",
            account_id=shop,
            account_name=shop,
        )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": "shopify api error", "detail": detail}), 502
    except URLError as exc:
        return jsonify({"error": "shopify network error", "detail": str(exc)}), 502

    return redirect("/admin?shopify=connected", code=302)


@app.post("/admin/publish/gumroad/<product_id>")
def admin_publish_gumroad(product_id: str):
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401

    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404

    try:
        listing = publish_product_to_gumroad(product)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": "gumroad api error", "detail": detail}), 502
    except URLError as exc:
        return jsonify({"error": "gumroad network error", "detail": str(exc)}), 502

    if flask_request.form.get("redirect") == "1":
        return redirect("/admin?gumroad=published", code=302)
    return jsonify({"ok": True, "product_id": product_id, "listing": listing})


@app.post("/admin/publish/gumroad-all")
def admin_publish_gumroad_all():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401

    products = list_products(active_only=True)
    results = []
    for product in products:
        try:
            listing = publish_product_to_gumroad(product)
            results.append({"product_id": product["id"], "title": product["title"], "ok": True, "listing": listing})
        except Exception as exc:
            results.append({"product_id": product["id"], "title": product["title"], "ok": False, "error": str(exc)})

    success_count = len([r for r in results if r["ok"]])
    fail_count = len(results) - success_count
    if flask_request.form.get("redirect") == "1":
        return redirect("/admin?gumroad=bulk", code=302)
    return jsonify({"ok": True, "success": success_count, "failed": fail_count, "results": results})


@app.post("/admin/publish/shopify/<product_id>")
def admin_publish_shopify(product_id: str):
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401

    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404

    try:
        listing = publish_product_to_shopify(product)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return jsonify({"error": "shopify api error", "detail": detail}), 502
    except URLError as exc:
        return jsonify({"error": "shopify network error", "detail": str(exc)}), 502

    if flask_request.form.get("redirect") == "1":
        return redirect("/admin?shopify=published", code=302)
    return jsonify({"ok": True, "product_id": product_id, "listing": listing})


@app.post("/admin/publish/shopify-all")
def admin_publish_shopify_all():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401

    products = list_products(active_only=True)
    results = []
    for product in products:
        try:
            listing = publish_product_to_shopify(product)
            results.append({"product_id": product["id"], "title": product["title"], "ok": True, "listing": listing})
        except Exception as exc:
            results.append({"product_id": product["id"], "title": product["title"], "ok": False, "error": str(exc)})

    success_count = len([r for r in results if r["ok"]])
    fail_count = len(results) - success_count
    if flask_request.form.get("redirect") == "1":
        return redirect("/admin?shopify=bulk", code=302)
    return jsonify({"ok": True, "success": success_count, "failed": fail_count, "results": results})


@app.get("/admin/leads")
def admin_leads():
    if not admin_guard():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"leads": list_recent_abandoned_leads(limit=50)})


@app.get("/admin/deliveries")
def admin_deliveries():
    if not admin_guard():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"deliveries": list_recent_deliveries(limit=50)})


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
    delivery_ok, delivery_note = handle_post_purchase_delivery(event, sale_id=sale_id)
    return jsonify(
        {
            "received": True,
            "sale_id": sale_id,
            "payout_id": payout_id,
            "delivery_sent": delivery_ok,
            "delivery_note": delivery_note,
        }
    )


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
