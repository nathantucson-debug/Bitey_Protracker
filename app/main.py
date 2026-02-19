import base64
import html
import hashlib
import hmac
import io
import json
import math
import os
import random
import re
import sqlite3
import smtplib
import sys
import threading
import time
import uuid
import wave
import zipfile
from array import array
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from urllib import parse, request
from urllib.error import HTTPError, URLError

from flask import Flask, Response, jsonify, redirect, render_template, request as flask_request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from protracker import protracker_bp

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
DOWNLOAD_LINK_SECRET = os.getenv("DOWNLOAD_LINK_SECRET", "")

PAYPAL_BASE = "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.register_blueprint(protracker_bp)


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
    override = get_content_override(product.get("title", ""))
    if override:
        details = {
            **details,
            "category": override.get("category") or details.get("category", "General"),
            "tagline": override.get("tagline") or details.get("tagline", ""),
            "description": override.get("description") or details.get("description", ""),
            "preview_items": override.get("preview_items") or details.get("preview_items", []),
            "preview_snippet": override.get("preview_snippet") or details.get("preview_snippet", ""),
        }
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


def download_secret() -> str:
    return DOWNLOAD_LINK_SECRET or STRIPE_WEBHOOK_SECRET or ADMIN_TOKEN


def create_download_token(payload: dict, ttl_seconds: int = 60 * 60 * 24 * 30) -> str:
    exp_ts = int(datetime.now(timezone.utc).timestamp()) + max(300, ttl_seconds)
    body = {**payload, "exp": exp_ts}
    body_b64 = b64url(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(download_secret().encode("utf-8"), body_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{body_b64}.{sig}"


def parse_download_token(token: str) -> dict | None:
    parts = (token or "").split(".", 1)
    if len(parts) != 2:
        return None
    body_b64, provided_sig = parts
    expected_sig = hmac.new(download_secret().encode("utf-8"), body_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, provided_sig):
        return None
    try:
        body = json.loads(b64url_decode(body_b64).decode("utf-8"))
    except Exception:
        return None
    exp = int(body.get("exp") or 0)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    if exp <= now_ts:
        return None
    return body


def _csv_escape(value: str) -> str:
    return str(value).replace('"', '""')


def _csv_line(values: list[str]) -> str:
    return ",".join(f'"{_csv_escape(v)}"' for v in values)


def _csv_block(headers: list[str], rows: list[list[str]]) -> str:
    lines = [_csv_line(headers)]
    lines.extend(_csv_line([str(c) for c in row]) for row in rows)
    return "\n".join(lines)


WAVE1_PRODUCT_ASSETS = {
    "Creator Caption Vault": {
        "buyer_result": "Publish 7 conversion-focused captions in one week with clear CTAs.",
        "quickstart_steps": [
            "Pick one offer and one audience segment.",
            "Customize the 7-day caption planner with your product details.",
            "Post day 1 and day 2 captions today.",
            "Track saves, profile visits, and DMs in the weekly tracker.",
        ],
        "scripts": [
            "Hook: If your content gets views but no sales, start here.",
            "Value: Use this 3-part caption format: problem, proof, next step.",
            "CTA: Comment READY and I will send the exact framework.",
            "Offer: Use this caption planner to publish consistently this week.",
        ],
        "assets": [
            {
                "name": "7-Day Caption Planner",
                "headers": ["Day", "Hook", "Core Message", "CTA", "Offer Mention", "Status"],
                "template_rows": [
                    ["Mon", "Call out the main pain point", "Teach one practical fix", "Save this post", "Soft", "Planned"],
                    ["Tue", "Share a common mistake", "Show your framework", "Comment GUIDE", "Medium", "Planned"],
                    ["Wed", "Debunk a myth", "Provide proof", "DM START", "Strong", "Planned"],
                ],
                "completed_rows": [
                    ["Mon", "Posting every day but still no leads?", "Your caption needs a clear next step, not more words.", "Save this for your next post", "Creator Ops Kit", "Completed"],
                    ["Tue", "Most creators skip this conversion step", "Use pain -> proof -> step in every post this week.", "Comment GUIDE", "Creator Ops Kit", "Completed"],
                    ["Wed", "Your CTA might be killing momentum", "One CTA per post outperforms mixed actions.", "DM START", "Creator Ops Kit", "Completed"],
                ],
                "purpose": "Plan and execute one full week of strategic captions.",
            },
            {
                "name": "CTA Matrix",
                "headers": ["Goal", "CTA Type", "Exact CTA", "Best Placement", "When To Use", "Status"],
                "template_rows": [
                    ["Engagement", "Comment", "Comment KEYWORD", "Last line", "Educational posts", "Planned"],
                    ["Lead capture", "DM", "DM START", "Last line + P.S.", "Offer posts", "Planned"],
                ],
                "completed_rows": [
                    ["Engagement", "Comment", "Comment MAP for the full checklist", "Last line", "Framework posts", "Completed"],
                    ["Lead capture", "DM", "DM START for the done-for-you version", "Last line + follow-up story", "Sales posts", "Completed"],
                ],
                "purpose": "Match CTA type to campaign goal so posts convert better.",
            },
            {
                "name": "Weekly Performance Review",
                "headers": ["Metric", "Target", "Actual", "Gap", "Action For Next Week", "Owner"],
                "template_rows": [
                    ["Saves", "120", "", "", "Improve value density in carousel posts", "You"],
                    ["Profile Visits", "80", "", "", "Add stronger CTA in day 2 and day 4 posts", "You"],
                ],
                "completed_rows": [
                    ["Saves", "120", "138", "+18", "Keep educational structure", "You"],
                    ["Profile Visits", "80", "74", "-6", "Test stronger CTA with DM keyword", "You"],
                ],
                "purpose": "Measure output quality and improve week-over-week.",
            },
        ],
    },
    "Short-Form Hook Library": {
        "buyer_result": "Ship 20 stronger hooks in one session for reels/shorts.",
        "assets": [
            {
                "name": "Hook Sprint Board",
                "headers": ["Theme", "Audience Pain", "Hook Draft", "Hook Type", "Platform", "Status"],
                "template_rows": [
                    ["Content strategy", "Low retention", "Use this pattern to keep people watching", "Pattern interrupt", "Reels", "Draft"],
                    ["Sales", "No inbound DMs", "If your videos get views but no buyers...", "Pain callout", "TikTok", "Draft"],
                ],
                "completed_rows": [
                    ["Content strategy", "Low retention", "Most creators lose viewers in the first 2 seconds. Fix this first.", "Pattern interrupt", "Reels", "Completed"],
                    ["Sales", "No inbound DMs", "Your content can be good and still not sell. Here is why.", "Pain callout", "TikTok", "Completed"],
                ],
                "purpose": "Generate and test hooks quickly by intent.",
            },
            {
                "name": "Retention Hook Scorecard",
                "headers": ["Hook", "Clarity (1-10)", "Curiosity (1-10)", "Specificity (1-10)", "Total", "Decision"],
                "template_rows": [
                    ["Most people do this wrong...", "", "", "", "", "Review"],
                    ["If you only fix one thing this week...", "", "", "", "", "Review"],
                ],
                "completed_rows": [
                    ["Most people do this wrong with short-form content", "8", "7", "7", "22", "Publish"],
                    ["If you only fix one thing this week, fix your opener", "9", "8", "9", "26", "Publish"],
                ],
                "purpose": "Score hooks objectively before publishing.",
            },
            {
                "name": "30-Day Hook Rotation",
                "headers": ["Week", "Hook Theme", "Primary CTA", "Publishing Slots", "Owner", "Status"],
                "template_rows": [
                    ["Week 1", "Pain + quick fix", "Save", "Mon/Wed/Fri", "You", "Planned"],
                    ["Week 2", "Myth vs reality", "Comment", "Mon/Wed/Fri", "You", "Planned"],
                ],
                "completed_rows": [
                    ["Week 1", "Pain + quick fix", "Save", "Mon/Wed/Fri", "You", "Completed"],
                    ["Week 2", "Myth vs reality", "Comment KEYWORD", "Mon/Wed/Fri", "You", "In Progress"],
                ],
                "purpose": "Turn hooks into a repeatable monthly system.",
            },
        ],
    },
    "UGC Pitch Deck Kit": {
        "buyer_result": "Send polished brand pitch decks that convert to discovery calls.",
        "assets": [
            {
                "name": "Brand Pitch Deck Outline",
                "headers": ["Slide", "Content Goal", "What To Include", "Proof Element", "CTA", "Status"],
                "template_rows": [
                    ["1", "Positioning", "Who you help and key outcome", "Niche experience", "Continue", "Draft"],
                    ["2", "Services", "Packages and deliverables", "Scope clarity", "Continue", "Draft"],
                ],
                "completed_rows": [
                    ["1", "Positioning", "I help skincare brands increase paid ad CTR with creator-led UGC.", "30+ short-form assets delivered", "Continue", "Completed"],
                    ["2", "Services", "Starter, Growth, and Scale package with timelines.", "Clear revision policy included", "Book a call", "Completed"],
                ],
                "purpose": "Build a credible deck structure before design polish.",
            },
            {
                "name": "Rate Card Builder",
                "headers": ["Package", "Deliverables", "Timeline", "Price", "Revision Policy", "Status"],
                "template_rows": [
                    ["Starter", "3 videos", "7 days", "", "1 revision", "Draft"],
                    ["Growth", "6 videos + hooks", "10 days", "", "2 revisions", "Draft"],
                ],
                "completed_rows": [
                    ["Starter", "3 videos", "7 days", "$650", "1 revision", "Completed"],
                    ["Growth", "6 videos + hooks", "10 days", "$1,150", "2 revisions", "Completed"],
                ],
                "purpose": "Create clear package pricing for faster approvals.",
            },
            {
                "name": "Outreach Follow-Up Sequence",
                "headers": ["Touch", "Timing", "Message Goal", "Script", "CTA", "Status"],
                "template_rows": [
                    ["1", "Day 0", "Initial value pitch", "", "Reply", "Draft"],
                    ["2", "Day 3", "Proof and reminder", "", "Book call", "Draft"],
                ],
                "completed_rows": [
                    ["1", "Day 0", "Initial value pitch", "Built a concept board for your upcoming launch with 3 hooks.", "Reply if you want the concept sample", "Completed"],
                    ["2", "Day 3", "Proof and reminder", "Recent campaign improved CTR by 28% in 2 weeks.", "Choose one of these call slots", "Completed"],
                ],
                "purpose": "Run outreach in a consistent cadence with proof.",
            },
        ],
    },
    "Notion Operator System": {
        "buyer_result": "Move from scattered tasks to a weekly operating rhythm.",
        "assets": [
            {
                "name": "Weekly Command Dashboard",
                "headers": ["Priority", "Task", "Time Block", "Owner", "Definition Of Done", "Status"],
                "template_rows": [
                    ["High", "Launch offer page update", "Mon 9:00-11:00", "You", "Page published", "Planned"],
                    ["High", "Client delivery batch", "Tue 10:00-12:00", "You", "All files delivered", "Planned"],
                ],
                "completed_rows": [
                    ["High", "Launch offer page update", "Mon 9:00-11:00", "You", "Page published", "Completed"],
                    ["High", "Client delivery batch", "Tue 10:00-12:00", "You", "All files delivered", "Completed"],
                ],
                "purpose": "Run each week from one outcome-first board.",
            },
            {
                "name": "Project Pipeline Tracker",
                "headers": ["Project", "Stage", "Next Action", "Owner", "Deadline", "Risk"],
                "template_rows": [
                    ["Spring campaign", "Planning", "Finalize messaging brief", "You", "Week 1", "Low"],
                    ["Client onboarding refresh", "Execution", "Publish SOP update", "Ops", "Week 2", "Medium"],
                ],
                "completed_rows": [
                    ["Spring campaign", "Execution", "Record launch assets", "You", "Week 1", "Low"],
                    ["Client onboarding refresh", "Review", "QA final flow", "Ops", "Week 2", "Low"],
                ],
                "purpose": "Track all active projects and unblock bottlenecks early.",
            },
            {
                "name": "Weekly Review Template",
                "headers": ["Section", "Prompt", "Your Notes", "Action For Next Week"],
                "template_rows": [
                    ["Wins", "What shipped this week?", "", ""],
                    ["Bottlenecks", "What slowed execution?", "", ""],
                ],
                "completed_rows": [
                    ["Wins", "What shipped this week?", "Published 3 offer pages and delivered 2 client batches.", "Maintain Tuesday/Thursday build blocks"],
                    ["Bottlenecks", "What slowed execution?", "Context switching across small admin tasks.", "Batch all admin into one afternoon block"],
                ],
                "purpose": "Close each week with data, not guesswork.",
            },
        ],
    },
    "Budget & Cashflow Spreadsheet Pro": {
        "buyer_result": "Know your true monthly runway and improve spending decisions.",
        "assets": [
            {
                "name": "Monthly Budget Dashboard",
                "headers": ["Category", "Planned", "Actual", "Variance", "Action", "Owner"],
                "template_rows": [
                    ["Revenue", "12000", "", "", "Update weekly", "You"],
                    ["Tools/Software", "450", "", "", "Trim unused subscriptions", "You"],
                ],
                "completed_rows": [
                    ["Revenue", "12000", "12750", "+750", "Keep same growth channel mix", "You"],
                    ["Tools/Software", "450", "520", "-70", "Cancel 2 unused tools", "You"],
                ],
                "purpose": "Track plan vs actual in a single monthly board.",
            },
            {
                "name": "8-Week Cashflow Forecast",
                "headers": ["Week", "Projected Inflow", "Projected Outflow", "Net", "Runway Note", "Status"],
                "template_rows": [
                    ["Week 1", "", "", "", "Maintain 6+ week runway", "Draft"],
                    ["Week 2", "", "", "", "Watch large renewal payments", "Draft"],
                ],
                "completed_rows": [
                    ["Week 1", "3,200", "2,450", "+750", "Runway stable", "Completed"],
                    ["Week 2", "2,900", "3,100", "-200", "Delay discretionary spend", "Completed"],
                ],
                "purpose": "Prevent cash surprises before they happen.",
            },
            {
                "name": "Profit Improvement Plan",
                "headers": ["Opportunity", "Current State", "Target State", "Impact", "Owner", "Deadline"],
                "template_rows": [
                    ["Reduce software overhead", "", "", "", "You", "Week 2"],
                    ["Increase average order value", "", "", "", "You", "Week 4"],
                ],
                "completed_rows": [
                    ["Reduce software overhead", "$520/month", "$390/month", "+$130/month margin", "You", "Week 2"],
                    ["Increase average order value", "$86", "$102", "+18% AOV", "You", "Week 4"],
                ],
                "purpose": "Convert financial insights into practical margin gains.",
            },
        ],
    },
    "Resume + Interview Kit": {
        "buyer_result": "Present measurable achievements and increase interview conversion.",
        "assets": [
            {
                "name": "Resume Impact Builder",
                "headers": ["Role Target", "Achievement Input", "Impact Bullet", "Metric", "Status"],
                "template_rows": [
                    ["Product Marketing Manager", "Improved onboarding flow", "", "", "Draft"],
                    ["Operations Lead", "Reduced project delays", "", "", "Draft"],
                ],
                "completed_rows": [
                    ["Product Marketing Manager", "Improved onboarding flow", "Improved activation rate by 18% through onboarding redesign.", "Activation +18%", "Completed"],
                    ["Operations Lead", "Reduced project delays", "Reduced average delivery delays by 35% by reworking planning cadence.", "Delay -35%", "Completed"],
                ],
                "purpose": "Turn experience into strong, quantifiable resume bullets.",
            },
            {
                "name": "Interview Story Bank",
                "headers": ["Question Type", "Situation", "Action", "Result", "Lesson", "Status"],
                "template_rows": [
                    ["Leadership", "", "", "", "", "Draft"],
                    ["Conflict", "", "", "", "", "Draft"],
                ],
                "completed_rows": [
                    ["Leadership", "Team missing deadlines", "Built weekly planning system", "Delivery speed improved 24%", "Simple operating rhythm wins", "Completed"],
                    ["Conflict", "Cross-team priority mismatch", "Set decision framework", "Escalations dropped 40%", "Clarify ownership early", "Completed"],
                ],
                "purpose": "Prepare polished, reusable interview answers.",
            },
            {
                "name": "Offer Follow-Up Kit",
                "headers": ["Scenario", "Email Objective", "Message Draft", "CTA", "Status"],
                "template_rows": [
                    ["Post-interview", "Thank interviewer and reinforce fit", "", "Reply", "Draft"],
                    ["Offer negotiation", "Align compensation with scope", "", "Schedule call", "Draft"],
                ],
                "completed_rows": [
                    ["Post-interview", "Thank interviewer and reinforce fit", "Thank you for the interview. I’m excited to bring structured launch execution to this role.", "Happy to share a 30-day plan", "Completed"],
                    ["Offer negotiation", "Align compensation with scope", "Based on scope and benchmark, I’d like to discuss a base of $X with the same start timeline.", "Can we review this together?", "Completed"],
                ],
                "purpose": "Handle post-interview communication professionally.",
            },
        ],
    },
    "Freelance Client Pack": {
        "buyer_result": "Onboard clients faster with fewer revisions and faster payments.",
        "assets": [
            {
                "name": "Client Scope Agreement",
                "headers": ["Section", "Clause", "Your Version", "Client-Friendly Summary", "Status"],
                "template_rows": [
                    ["Deliverables", "List included outputs", "", "", "Draft"],
                    ["Revisions", "Define number of rounds", "", "", "Draft"],
                ],
                "completed_rows": [
                    ["Deliverables", "3 landing page sections + 1 revision round", "3 sections delivered in 5 business days", "You receive launch-ready copy in one week", "Completed"],
                    ["Revisions", "Two revision rounds included", "Feedback window: 72 hours", "Clear edits with predictable turnaround", "Completed"],
                ],
                "purpose": "Prevent scope creep with explicit written terms.",
            },
            {
                "name": "Onboarding Intake Form",
                "headers": ["Question", "Why It Matters", "Client Response", "Owner", "Status"],
                "template_rows": [
                    ["What is your primary business goal this month?", "Align copy with priority outcome", "", "Client", "Open"],
                    ["Who is the exact target buyer?", "Improve conversion clarity", "", "Client", "Open"],
                ],
                "completed_rows": [
                    ["What is your primary business goal this month?", "Align copy with priority outcome", "Increase consult bookings by 20%", "Client", "Complete"],
                    ["Who is the exact target buyer?", "Improve conversion clarity", "US-based service businesses with 2-10 team members", "Client", "Complete"],
                ],
                "purpose": "Collect all project-critical details before kickoff.",
            },
            {
                "name": "Invoice + Follow-Up Tracker",
                "headers": ["Invoice", "Amount", "Due Date", "Status", "Follow-Up Script", "Owner"],
                "template_rows": [
                    ["INV-001", "", "Week 1", "Sent", "", "You"],
                    ["INV-002", "", "Week 2", "Draft", "", "You"],
                ],
                "completed_rows": [
                    ["INV-001", "$750", "Week 1", "Paid", "Thanks for the quick payment.", "You"],
                    ["INV-002", "$750", "Week 2", "Sent", "Friendly reminder: final invoice due tomorrow.", "You"],
                ],
                "purpose": "Keep payment collection organized and professional.",
            },
        ],
    },
    "Wedding Invite Suite (Canva)": {
        "buyer_result": "Produce a polished, cohesive wedding stationery suite quickly.",
        "assets": [
            {
                "name": "Invitation Copy Master",
                "headers": ["Section", "Required Info", "Draft Copy", "Final Copy", "Status"],
                "template_rows": [
                    ["Main invite", "Couple names, date, venue", "", "", "Draft"],
                    ["RSVP", "Deadline and response method", "", "", "Draft"],
                ],
                "completed_rows": [
                    ["Main invite", "Couple names, date, venue", "Join us to celebrate...", "Together with their families, Anna and Marco invite you...", "Completed"],
                    ["RSVP", "Deadline and response method", "Please RSVP by June 15", "Please RSVP by June 15 at ourwedding.com/rsvp", "Completed"],
                ],
                "purpose": "Finalize invitation text before visual customization.",
            },
            {
                "name": "Print Export Checklist",
                "headers": ["Asset", "Size Verified", "Bleed Added", "Color Mode", "Exported PDF", "Status"],
                "template_rows": [
                    ["Invitation front", "No", "No", "RGB", "No", "Draft"],
                    ["RSVP card", "No", "No", "RGB", "No", "Draft"],
                ],
                "completed_rows": [
                    ["Invitation front", "Yes", "Yes", "CMYK", "Yes", "Completed"],
                    ["RSVP card", "Yes", "Yes", "CMYK", "Yes", "Completed"],
                ],
                "purpose": "Avoid costly print mistakes before sending to vendor.",
            },
            {
                "name": "Guest Communication Timeline",
                "headers": ["Milestone", "Message", "Delivery Channel", "Timing", "Owner", "Status"],
                "template_rows": [
                    ["Save the date", "Initial announcement", "Email", "8 months before", "Couple", "Planned"],
                    ["Reminder", "RSVP reminder", "Text/Email", "2 weeks before deadline", "Couple", "Planned"],
                ],
                "completed_rows": [
                    ["Save the date", "Save our date: Sept 12", "Email", "8 months before", "Couple", "Completed"],
                    ["Reminder", "Friendly RSVP reminder", "Email", "2 weeks before deadline", "Couple", "Completed"],
                ],
                "purpose": "Keep guest communication clear and on schedule.",
            },
        ],
    },
    "Airbnb Host Welcome Book": {
        "buyer_result": "Reduce repetitive guest messages and improve review quality.",
        "assets": [
            {
                "name": "Guest Welcome Script Pack",
                "headers": ["Stage", "Goal", "Message Template", "Send Timing", "Owner", "Status"],
                "template_rows": [
                    ["Pre-arrival", "Set clear expectations", "", "24h before check-in", "Host", "Draft"],
                    ["Arrival", "Smooth check-in", "", "Check-in time", "Host", "Draft"],
                ],
                "completed_rows": [
                    ["Pre-arrival", "Set clear expectations", "Welcome! Here are parking, code, and Wi-Fi details before you arrive.", "24h before check-in", "Host", "Completed"],
                    ["Arrival", "Smooth check-in", "Hope check-in went smoothly. Reply here if you need anything.", "Check-in time", "Host", "Completed"],
                ],
                "purpose": "Standardize communication across the full guest journey.",
            },
            {
                "name": "House Rules + Essentials Card",
                "headers": ["Section", "Rule/Info", "Why It Matters", "Guest-Friendly Wording", "Status"],
                "template_rows": [
                    ["Quiet hours", "", "Respect neighbors", "", "Draft"],
                    ["Checkout", "", "Faster turnover", "", "Draft"],
                ],
                "completed_rows": [
                    ["Quiet hours", "10 PM to 8 AM", "Respect neighbors", "Please keep noise low after 10 PM.", "Completed"],
                    ["Checkout", "Start dishwasher + lock door", "Faster turnover", "Before leaving, please start dishwasher and lock the door.", "Completed"],
                ],
                "purpose": "Publish clear rules that reduce confusion and support requests.",
            },
            {
                "name": "Issue Resolution Tracker",
                "headers": ["Issue Type", "Guest Message", "Response Script", "ETA", "Resolution", "Status"],
                "template_rows": [
                    ["Wi-Fi", "", "", "15 min", "", "Open"],
                    ["Temperature", "", "", "20 min", "", "Open"],
                ],
                "completed_rows": [
                    ["Wi-Fi", "Wi-Fi is down", "Thanks for flagging. I reset the router and will confirm in 10 minutes.", "10 min", "Resolved with reset", "Completed"],
                    ["Temperature", "AC seems weak", "Thank you. I adjusted settings and arranged backup fan delivery.", "20 min", "Guest confirmed comfort", "Completed"],
                ],
                "purpose": "Respond quickly and consistently to common guest issues.",
            },
        ],
    },
    "Etsy Listing SEO Toolkit": {
        "buyer_result": "Improve listing discoverability and conversion quality.",
        "assets": [
            {
                "name": "Listing Optimization Builder",
                "headers": ["Listing", "Primary Keyword", "SEO Title", "Opening Description", "Primary CTA", "Status"],
                "template_rows": [
                    ["Product A", "printable planner", "", "", "", "Draft"],
                    ["Product B", "wedding invite template", "", "", "", "Draft"],
                ],
                "completed_rows": [
                    ["Product A", "printable planner", "Printable Planner Bundle for Busy Professionals", "Organize your week with a printable planner that is easy to customize.", "Add to cart", "Completed"],
                    ["Product B", "wedding invite template", "Editable Wedding Invite Template (Canva)", "Create elegant wedding invitations in minutes with an editable Canva file.", "Download now", "Completed"],
                ],
                "purpose": "Build search-intent titles and conversion-focused first lines.",
            },
            {
                "name": "Tag Strategy Worksheet",
                "headers": ["Listing", "Tag 1", "Tag 2", "Tag 3", "Tag 4", "Tag 5"],
                "template_rows": [
                    ["Product A", "", "", "", "", ""],
                    ["Product B", "", "", "", "", ""],
                ],
                "completed_rows": [
                    ["Product A", "printable planner", "daily planner pdf", "weekly planner", "digital planner", "productivity printable"],
                    ["Product B", "wedding invite template", "editable invitation", "canva wedding invite", "printable wedding", "digital wedding invite"],
                ],
                "purpose": "Fill all high-intent tags with relevance and variety.",
            },
            {
                "name": "Photo + Thumbnail Checklist",
                "headers": ["Image Slot", "Image Type", "Message Goal", "Overlay Text", "Status"],
                "template_rows": [
                    ["1", "Cover image", "Instant clarity", "", "Draft"],
                    ["2", "Feature callout", "What is included", "", "Draft"],
                ],
                "completed_rows": [
                    ["1", "Cover image", "Instant clarity", "Editable Canva Template", "Completed"],
                    ["2", "Feature callout", "What is included", "Includes invitation + RSVP + details card", "Completed"],
                ],
                "purpose": "Create thumbnails that drive clicks and reduce buyer confusion.",
            },
        ],
    },
}


def _product_content_profile(product: dict) -> dict:
    title = product.get("title", "Digital Product")
    category = product.get("category", "General")
    preview = product.get("real_world_preview", {})
    included = product.get("preview_items", [])
    rows = preview.get("rows") or []

    category_profiles = {
        "Creator Growth": {
            "buyer_goal": "publish conversion-focused content consistently",
            "quick_win": "use one hook + one proof + one CTA to publish your first post today",
            "scripts": [
                "Hook: Most creators are stuck because they skip this one step.",
                "Story: Here is exactly what changed results for my audience.",
                "CTA: Comment GUIDE and I will send the full template.",
                "Offer line: If you want this system done-for-you, reply READY.",
            ],
            "kpis": ["Posts Published", "Average Watch Time", "Saves", "Profile Clicks", "DM Starts"],
        },
        "Creator Business": {
            "buyer_goal": "close better client opportunities with clearer positioning",
            "quick_win": "customize the package/pricing section and send one pitch today",
            "scripts": [
                "Pitch opener: I create conversion-focused short-form assets for [brand type].",
                "Proof line: Recent campaigns improved [metric] by [result].",
                "Package line: I recommend starting with the [Starter] test package.",
                "Close: Would you like a tailored concept board for your next launch?",
            ],
            "kpis": ["Pitches Sent", "Reply Rate", "Calls Booked", "Proposal Acceptance", "Revenue Closed"],
        },
        "Ecommerce": {
            "buyer_goal": "increase product page discoverability and conversion",
            "quick_win": "optimize one listing title, tags, and first description paragraph",
            "scripts": [
                "Title format: Core keyword + specific use case + differentiator.",
                "Opening line: Buyer outcome in one sentence, no fluff.",
                "Feature bullets: Problem solved, key specs, what is included.",
                "Conversion CTA: Add to cart now to get instant access.",
            ],
            "kpis": ["Impressions", "CTR", "Favorites", "Add To Cart Rate", "Orders"],
        },
        "Operations": {
            "buyer_goal": "standardize repeatable workflows with less rework",
            "quick_win": "document one critical SOP and run it once this week",
            "scripts": [
                "SOP purpose: This process ensures quality and predictable turnaround.",
                "Trigger: Start this SOP when [event] occurs.",
                "QA check: Confirm output meets [specific acceptance criteria].",
                "Escalation: If blocked, route to [owner] within [SLA].",
            ],
            "kpis": ["Cycle Time", "Missed Steps", "Rework Rate", "On-Time Delivery", "Owner Clarity Score"],
        },
        "Productivity": {
            "buyer_goal": "plan and execute priorities without chaos",
            "quick_win": "set top 3 outcomes for this week and time-block them now",
            "scripts": [
                "Daily plan: Top outcome, deep-work block, admin block, review.",
                "Decision rule: If it does not move weekly goals, defer it.",
                "Shutdown ritual: Capture open loops and assign next action.",
                "Weekly review: Keep, improve, remove one workflow.",
            ],
            "kpis": ["Priority Completion", "Deep Work Hours", "Carryover Tasks", "Blocked Tasks", "Weekly Wins"],
        },
        "Finance": {
            "buyer_goal": "improve cash clarity and spending decisions",
            "quick_win": "enter this month’s top 10 transactions and categorize them",
            "scripts": [
                "Revenue check: Compare actual vs planned every Friday.",
                "Expense check: Flag categories above budget by 10%+.",
                "Cashflow note: Confirm runway for next 8 weeks.",
                "Action trigger: Cut, hold, or scale based on margin threshold.",
            ],
            "kpis": ["Monthly Revenue", "Net Margin", "Burn Rate", "Cash Runway (Weeks)", "Savings Rate"],
        },
        "Career": {
            "buyer_goal": "position achievements for interviews and offers",
            "quick_win": "rewrite your top 3 resume bullets using metrics today",
            "scripts": [
                "Resume bullet: Action + measurable result + business impact.",
                "Cover line: Why this role, why now, why you.",
                "Interview story: Situation, action, outcome, lesson.",
                "Follow-up: Thank you + concise value recap + next-step ask.",
            ],
            "kpis": ["Applications Sent", "Callbacks", "Interviews", "Final Rounds", "Offers"],
        },
        "Freelance Ops": {
            "buyer_goal": "reduce scope creep and improve payment reliability",
            "quick_win": "send updated onboarding + scope doc to one active client",
            "scripts": [
                "Scope clause: Includes X deliverables and Y revision rounds.",
                "Timeline clause: Delivery date assumes feedback in 48 hours.",
                "Payment clause: 50% upfront, 50% before final handoff.",
                "Boundary script: New request can be added as phase two.",
            ],
            "kpis": ["Project Margin", "Average Days To Pay", "Scope Creep Incidents", "Client NPS", "Referrals"],
        },
        "Wellness": {
            "buyer_goal": "stick to a repeatable healthy routine",
            "quick_win": "plan your next 3 days of meals/training in one pass",
            "scripts": [
                "Weekly plan: Anchor sessions first, then add support blocks.",
                "Habit cue: Pair new habit with an existing routine.",
                "Fallback plan: Minimum viable version for busy days.",
                "Review note: Keep what worked, adjust one friction point.",
            ],
            "kpis": ["Planned Sessions", "Completed Sessions", "Streak Days", "Energy Score", "Prep Consistency"],
        },
        "Family": {
            "buyer_goal": "run calmer household routines with clear expectations",
            "quick_win": "fill this week’s schedule and post it in one visible place",
            "scripts": [
                "Routine prompt: Here is what happens before school and bedtime.",
                "Reward rule: Points unlock specific rewards, not random treats.",
                "Family check-in: What worked this week and what felt hard?",
                "Reset line: Missed days are normal; restart at the next block.",
            ],
            "kpis": ["Routine Completion", "Task Completion", "Missed Tasks", "Family Stress Score", "Weekly Wins"],
        },
        "Events": {
            "buyer_goal": "ship polished event assets without last-minute stress",
            "quick_win": "customize names/date/style in the main template today",
            "scripts": [
                "Invite copy: Event details + tone + RSVP deadline in one screen.",
                "Reminder text: Friendly nudge with clear date/time and link.",
                "Vendor note: Confirm timeline, quantities, and setup windows.",
                "Day-of checklist: Final print/export review before release.",
            ],
            "kpis": ["Assets Completed", "Revisions Needed", "On-Time Milestones", "Vendor Confirmations", "Guest Response Rate"],
        },
        "Hospitality": {
            "buyer_goal": "deliver smoother guest experiences with fewer support requests",
            "quick_win": "publish your welcome + check-in + checkout instructions now",
            "scripts": [
                "Welcome message: Key access info + quick essentials.",
                "Issue response: Acknowledge issue, action ETA, follow-up plan.",
                "Local recommendation: Top 3 picks by guest intent.",
                "Checkout reminder: concise task list + appreciation note.",
            ],
            "kpis": ["Guest Questions", "Issue Resolution Time", "Review Rating", "Repeat Bookings", "Host Time Saved"],
        },
        "Branding": {
            "buyer_goal": "present a consistent visual/message identity",
            "quick_win": "apply your brand colors/fonts to one customer-facing asset",
            "scripts": [
                "Brand promise: We help [audience] achieve [outcome] without [pain].",
                "Voice guideline: Keep copy clear, direct, and outcome-first.",
                "CTA formula: Benefit + action + confidence statement.",
                "Consistency rule: One headline style across all channels.",
            ],
            "kpis": ["Asset Consistency Score", "Content Output", "Engagement Rate", "Brand Recall", "Lead Conversion"],
        },
        "Real Estate": {
            "buyer_goal": "convert property leads into qualified conversations",
            "quick_win": "send the lead follow-up sequence to one new prospect",
            "scripts": [
                "Lead welcome: deliver value asset + clear next step.",
                "Qualification prompt: budget, timeline, location priorities.",
                "Follow-up cadence: day 0, day 2, day 5, day 8.",
                "Consult close: offer two specific meeting options.",
            ],
            "kpis": ["New Leads", "Response Rate", "Consults Booked", "Showings", "Closed Deals"],
        },
    }
    defaults = category_profiles.get(
        category,
        {
            "buyer_goal": "implement a practical workflow that produces measurable outcomes",
            "quick_win": "complete the first implementation pass today",
            "scripts": [
                "Outcome: Define success criteria before editing templates.",
                "Implementation: Fill the template with your real business context.",
                "Quality check: Validate output against the provided checklist.",
                "Optimization: Improve one bottleneck each week.",
            ],
            "kpis": ["Output Volume", "Completion Rate", "Cycle Time", "Quality Score", "Buyer Satisfaction"],
        },
    )

    plan_rows = []
    for i, item in enumerate(included[:4], start=1):
        plan_rows.append(
            [
                f"Step {i}",
                f"Complete {item}",
                f"Customize and finalize the {item.lower()} file for your business use case.",
                "Today" if i == 1 else f"Day {i}",
                f"{item} ready for real-world use",
            ]
        )
    if len(plan_rows) < 4:
        for idx, row in enumerate(rows[: max(0, 4 - len(plan_rows))], start=len(plan_rows) + 1):
            name = row[0] if len(row) > 0 else f"Deliverable {idx}"
            action = row[1] if len(row) > 1 else "Finalize the deliverable"
            plan_rows.append(
                [
                    f"Step {idx}",
                    str(name),
                    f"{action}. Complete and save your final version.",
                    f"Day {idx}",
                    "Finished version exported",
                ]
            )
    if not plan_rows:
        plan_rows = [
            ["Step 1", "Customize core file", "Replace sample text with your real business details.", "Today", "Buyer-ready first draft"],
            ["Step 2", "Publish or send", "Use the completed file in your live workflow immediately.", "Day 2", "First live usage completed"],
            ["Step 3", "Improve version", "Apply feedback and finalize your polished version.", "Day 3", "Final version complete"],
        ]

    workbook_rows = []
    for idx, row in enumerate(plan_rows, start=1):
        workbook_rows.append([f"W{idx:02d}", row[1], row[2], row[3], "Not Started", row[4], ""])

    quickstart_steps = [f"{row[1]}: {row[2]}" for row in plan_rows[:4]]
    quickstart_steps.append(f"Run a full pass and compare results to: {preview.get('result', 'your target outcome')}")

    kpi_rows = [[metric, "", "", "", "", "", "", ""] for metric in defaults["kpis"]]
    calendar_rows = []
    for day in range(1, 31):
        source = plan_rows[(day - 1) % len(plan_rows)]
        calendar_rows.append([f"Day {day}", source[1], source[4], "Not Started", ""])

    return {
        "buyer_goal": defaults["buyer_goal"],
        "quick_win": defaults["quick_win"],
        "scripts": defaults["scripts"],
        "plan_rows": plan_rows,
        "workbook_rows": workbook_rows,
        "quickstart_steps": quickstart_steps,
        "kpi_rows": kpi_rows,
        "calendar_rows": calendar_rows,
    }


def _category_asset_blueprint(category: str) -> dict:
    blueprints = {
        "Creator Growth": {
            "core_headers": ["Content Type", "Hook", "Value Angle", "CTA", "Publishing Day", "Status"],
            "core_rows": [
                ["Reel", "Stop wasting hours on random posts", "3-step planning framework", "Comment GUIDE", "Monday", "Ready"],
                ["Carousel", "Most creators miss this retention fix", "Pattern interrupt + proof", "Save this", "Wednesday", "Ready"],
                ["Email", "A simple content engine for busy weeks", "Repurpose workflow", "Reply PLAN", "Friday", "Draft"],
            ],
            "script_bank": [
                "Hook: Most creators are overcomplicating growth.",
                "Bridge: Here is the framework I use with clients.",
                "Value: Step 1, Step 2, Step 3 in plain language.",
                "CTA: Save this and use it for your next post.",
            ],
        },
        "Creator Business": {
            "core_headers": ["Offer", "Deliverables", "Price", "Timeline", "Buyer Outcome", "Status"],
            "core_rows": [
                ["Starter Package", "3 assets + revision round", "$490", "5 days", "Fast launch support", "Ready"],
                ["Growth Package", "6 assets + messaging doc", "$990", "7 days", "Higher conversion", "Ready"],
                ["Premium Package", "12 assets + strategy call", "$1890", "14 days", "End-to-end implementation", "Draft"],
            ],
            "script_bank": [
                "Pitch opener: I help [audience] achieve [outcome] without [pain].",
                "Proof line: Last launch improved [metric] by [result].",
                "Offer line: I recommend starting with the Growth package.",
                "Close: Want me to send a tailored one-page proposal?",
            ],
        },
        "Ecommerce": {
            "core_headers": ["Listing", "Primary Keyword", "Title Draft", "First Description Line", "CTA", "Status"],
            "core_rows": [
                ["Product 1", "wedding invite template", "Editable Wedding Invite Template", "Customize in minutes with Canva.", "Add to cart", "Ready"],
                ["Product 2", "etsy seo toolkit", "Etsy SEO Toolkit for Digital Sellers", "Improve ranking and conversion with structured tags.", "Download now", "Ready"],
                ["Product 3", "shopify product copy", "Shopify Product Description Swipe File", "Write clear, buyer-focused descriptions fast.", "Use this template", "Draft"],
            ],
            "script_bank": [
                "Title pattern: Keyword + use case + format.",
                "Description opener: Outcome first, feature second.",
                "Bullet style: Keep each bullet tied to buyer benefit.",
                "CTA: Buy now for instant access and implementation.",
            ],
        },
        "Finance": {
            "core_headers": ["Category", "Planned", "Actual", "Variance", "Action", "Owner"],
            "core_rows": [
                ["Revenue", "8500", "0", "0", "Update weekly", "You"],
                ["Operating Costs", "2400", "0", "0", "Track receipts", "You"],
                ["Savings", "1200", "0", "0", "Auto transfer", "You"],
                ["Debt Paydown", "600", "0", "0", "Schedule payment", "You"],
            ],
            "script_bank": [
                "Weekly review: check planned vs actual every Friday.",
                "Variance trigger: investigate any category above 10%.",
                "Cashflow rule: prioritize runway over vanity spending.",
                "Decision note: cut, hold, or scale each major expense.",
            ],
        },
        "Career": {
            "core_headers": ["Role Target", "Core Achievement", "Resume Bullet", "Interview Story", "Application Date", "Status"],
            "core_rows": [
                ["Growth Marketing Manager", "Increased lead quality 31%", "Increased qualified leads 31% in 2 quarters", "Campaign optimization story", "YYYY-MM-DD", "Ready"],
                ["Operations Lead", "Reduced cycle time 42%", "Reduced client onboarding cycle from 12 to 7 days", "Process redesign story", "YYYY-MM-DD", "Ready"],
                ["Product Marketing Manager", "Lifted activation 18%", "Improved activation rate by 18% via onboarding revamp", "Cross-team execution story", "YYYY-MM-DD", "Draft"],
            ],
            "script_bank": [
                "Resume formula: action + metric + business impact.",
                "Interview structure: situation, action, result, reflection.",
                "Negotiation opener: based on scope and market benchmark.",
                "Follow-up line: concise value recap + clear next step.",
            ],
        },
        "Freelance Ops": {
            "core_headers": ["Client", "Scope", "Timeline", "Payment Terms", "Revision Policy", "Status"],
            "core_rows": [
                ["Client A", "Landing page copy + edits", "5 business days", "50% upfront / 50% before handoff", "2 rounds", "Ready"],
                ["Client B", "Email sequence setup", "7 business days", "50% upfront / 50% at delivery", "2 rounds", "Ready"],
                ["Client C", "Sales page framework", "10 business days", "Milestone billing", "3 rounds", "Draft"],
            ],
            "script_bank": [
                "Scope line: includes X deliverables and Y revisions.",
                "Timeline line: timeline assumes feedback in 48 hours.",
                "Payment line: final assets released on final payment.",
                "Boundary line: additional requests move to phase two.",
            ],
        },
        "Wellness": {
            "core_headers": ["Day", "Primary Focus", "Session", "Duration", "Nutrition Focus", "Status"],
            "core_rows": [
                ["Monday", "Strength", "Upper body session", "45m", "Protein + hydration", "Ready"],
                ["Tuesday", "Recovery", "Walk + mobility", "30m", "Whole-food meals", "Ready"],
                ["Wednesday", "Strength", "Lower body session", "45m", "Meal prep block", "Ready"],
                ["Thursday", "Conditioning", "Intervals", "25m", "Balanced carbs", "Draft"],
            ],
            "script_bank": [
                "Plan anchor: lock workouts first, then fit meals.",
                "Fallback rule: use 20-minute minimum session on busy days.",
                "Consistency note: perfection is not required for progress.",
                "Weekly review: keep wins, adjust one friction point only.",
            ],
        },
        "Family": {
            "core_headers": ["Day", "Routine Block", "Task", "Owner", "Completion Rule", "Status"],
            "core_rows": [
                ["Monday", "Morning", "Breakfast + school prep", "Family", "Done by 7:45 AM", "Ready"],
                ["Monday", "Evening", "Homework + tidy", "Kids + Parent", "Done before 8:00 PM", "Ready"],
                ["Tuesday", "Morning", "Checklist repeat", "Family", "No missed steps", "Ready"],
                ["Friday", "Rewards", "Point review + reward choice", "Parent", "Weekly closeout complete", "Draft"],
            ],
            "script_bank": [
                "Routine cue: here is what happens next.",
                "Reward rule: points unlock pre-agreed rewards.",
                "Reset line: missed day is fine, restart at next block.",
                "Check-in prompt: what felt easy, what needs adjustment?",
            ],
        },
        "Events": {
            "core_headers": ["Milestone", "Deliverable", "Owner", "Due Date", "Status", "Notes"],
            "core_rows": [
                ["Planning", "Finalize style + wording", "You", "YYYY-MM-DD", "Ready", ""],
                ["Production", "Export print files", "You", "YYYY-MM-DD", "Ready", ""],
                ["Distribution", "Send invitations/reminders", "You", "YYYY-MM-DD", "Ready", ""],
                ["Day-of", "Run-of-show packet", "You", "YYYY-MM-DD", "Draft", ""],
            ],
            "script_bank": [
                "Invite copy: include date, time, location, RSVP deadline.",
                "Reminder copy: friendly nudge + clear action.",
                "Vendor message: confirm quantities, timing, setup window.",
                "Day-of note: one source of truth for all stakeholders.",
            ],
        },
        "Hospitality": {
            "core_headers": ["Guest Stage", "Message", "Delivery Channel", "Timing", "Expected Outcome", "Status"],
            "core_rows": [
                ["Pre-arrival", "Welcome + check-in instructions", "Message app", "24h before", "Fewer check-in questions", "Ready"],
                ["Arrival", "Wi-Fi + essentials quick card", "Printed + digital", "Check-in time", "Faster guest setup", "Ready"],
                ["Stay", "Local recommendations", "Guidebook", "Day 1", "Higher experience rating", "Ready"],
                ["Checkout", "Simple checkout checklist", "Message app", "Night before", "Cleaner turnover", "Draft"],
            ],
            "script_bank": [
                "Welcome line: we’re excited to host you.",
                "Issue line: thanks for flagging this, here is our ETA.",
                "Local guide line: our top picks by time of day.",
                "Checkout line: short checklist + thank-you note.",
            ],
        },
        "Operations": {
            "core_headers": ["Process", "Step", "Owner", "SLA", "Quality Check", "Status"],
            "core_rows": [
                ["Onboarding", "Kickoff + intake", "Ops", "24h", "All required fields complete", "Ready"],
                ["Delivery", "Create + QA output", "Ops", "48h", "Checklist pass", "Ready"],
                ["Review", "Approval + revisions", "Ops", "24h", "Stakeholder sign-off", "Ready"],
                ["Handoff", "Archive + report", "Ops", "24h", "Final files delivered", "Draft"],
            ],
            "script_bank": [
                "Trigger: run SOP when project moves to this stage.",
                "QA line: verify each acceptance criterion before handoff.",
                "Escalation line: blocked items routed within 2 hours.",
                "Closeout line: archive and report key lessons weekly.",
            ],
        },
        "Productivity": {
            "core_headers": ["Priority", "Task", "Block", "Owner", "Outcome", "Status"],
            "core_rows": [
                ["High", "Core outcome #1", "9:00-10:30", "You", "Shipped", "Ready"],
                ["High", "Core outcome #2", "11:00-12:00", "You", "In review", "Ready"],
                ["Medium", "Admin and follow-ups", "2:00-3:00", "You", "Inbox zero", "Ready"],
                ["Medium", "Weekly review", "Friday 4:00", "You", "Next week planned", "Draft"],
            ],
            "script_bank": [
                "Daily start: define top 3 outcomes before opening inbox.",
                "Focus rule: one task per block, no context switching.",
                "Shutdown rule: capture open loops and assign next action.",
                "Weekly review: keep, improve, remove one workflow.",
            ],
        },
        "Branding": {
            "core_headers": ["Brand Element", "Primary Rule", "Usage Example", "Channel", "Owner", "Status"],
            "core_rows": [
                ["Voice", "Clear, direct, warm", "Outcome-first headlines", "Website", "Marketing", "Ready"],
                ["Typography", "Sora + Manrope", "Consistent heading hierarchy", "All channels", "Design", "Ready"],
                ["Color", "Primary + accent usage", "Buttons + highlights", "Website", "Design", "Ready"],
                ["Messaging", "One core promise", "Hero section + product cards", "Storefront", "Marketing", "Draft"],
            ],
            "script_bank": [
                "Brand promise: We help [audience] get [outcome] with less friction.",
                "Headline rule: clear benefit in 6-10 words.",
                "CTA rule: one action per section.",
                "Consistency rule: same tone across product pages and emails.",
            ],
        },
        "Real Estate": {
            "core_headers": ["Lead Stage", "Message Goal", "Template", "Timing", "CTA", "Status"],
            "core_rows": [
                ["New lead", "Deliver value + trust", "Welcome + market guide", "Immediately", "Book consult", "Ready"],
                ["Warming", "Qualify need", "Budget + timeline prompts", "+2 days", "Reply with criteria", "Ready"],
                ["Active", "Move to showing", "Property shortlist message", "+5 days", "Pick showing slots", "Ready"],
                ["Closing", "Reduce friction", "Offer support checklist", "+8 days", "Confirm next step", "Draft"],
            ],
            "script_bank": [
                "Intro: thanks for reaching out, here is your guide.",
                "Qualification: what is your timeline and target area?",
                "Nurture: here are top 3 listings matching your criteria.",
                "Close: want two showing options for this week?",
            ],
        },
    }
    return blueprints.get(
        category,
        {
            "core_headers": ["Asset", "Purpose", "Primary Action", "Timing", "Outcome", "Status"],
            "core_rows": [
                ["Core template", "First implementation", "Customize with your details", "Today", "Ready-to-use output", "Ready"],
                ["Execution sheet", "Track completion", "Run one full pass", "This week", "First outcome recorded", "Ready"],
                ["Optimization note", "Refine quality", "Apply one improvement", "Week 2", "Better performance", "Draft"],
            ],
            "script_bank": [
                "Define a clear outcome before you customize files.",
                "Complete one full pass before optimizing details.",
                "Track weekly metrics to measure quality and consistency.",
                "Save your final version as the reusable baseline.",
            ],
        },
    )


def _guided_experience_html(product: dict) -> str:
    title = product.get("title", "Digital Product")
    category = product.get("category", "General")
    profile = _product_content_profile(product)
    blueprint = _category_asset_blueprint(category)
    data = {
        "title": title,
        "category": category,
        "tagline": product.get("tagline", ""),
        "description": product.get("description", ""),
        "included": product.get("preview_items", []),
        "plan_rows": profile.get("plan_rows", []),
        "scripts": blueprint.get("script_bank", []),
    }
    payload = json.dumps(data)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} - Guided Experience</title>
<style>
body{{margin:0;font-family:Manrope,Segoe UI,sans-serif;background:#eff4ff;color:#0f172a}}
.wrap{{max-width:1100px;margin:0 auto;padding:22px}}
.hero{{background:#fff;border:1px solid #d8e2f5;border-radius:16px;padding:18px 20px;box-shadow:0 14px 30px rgba(15,23,42,.08)}}
.hero h1{{margin:0 0 8px;font-size:2rem;font-family:Sora,Manrope,sans-serif}}
.muted{{color:#475569}}
.pill{{display:inline-block;font-size:.78rem;padding:5px 10px;border-radius:999px;background:#edf3ff;border:1px solid #d4e0ff;color:#1e40af;font-weight:700}}
.grid{{display:grid;grid-template-columns:260px 1fr;gap:14px;margin-top:14px}}
.side,.main{{background:#fff;border:1px solid #d8e2f5;border-radius:16px;padding:16px;box-shadow:0 10px 24px rgba(15,23,42,.06)}}
.step{{padding:10px;border:1px solid #dbe5f8;border-radius:10px;margin-bottom:8px;font-weight:700;background:#f8fbff}}
.section{{margin-bottom:16px;padding:12px;border:1px solid #e3ebfb;border-radius:12px}}
.section h3{{margin:0 0 8px;font-family:Sora,Manrope,sans-serif}}
label{{display:block;font-size:.85rem;font-weight:700;margin:8px 0 4px;color:#334155}}
input,textarea,select{{width:100%;border:1px solid #cfdaf1;border-radius:10px;padding:10px;font:inherit}}
textarea{{min-height:90px}}
table{{width:100%;border-collapse:collapse;font-size:.93rem}}
th,td{{border:1px solid #e5edfb;padding:8px;vertical-align:top;text-align:left}}
th{{background:#f8fbff}}
td[contenteditable="true"]{{background:#ffffff}}
.row{{display:flex;gap:10px;flex-wrap:wrap}}
button{{border:1px solid #1d4ed8;background:#1d4ed8;color:#fff;padding:10px 14px;border-radius:10px;font-weight:700;cursor:pointer}}
button.alt{{background:#fff;color:#0f172a;border-color:#cdd8f1}}
.box{{padding:10px;border:1px solid #d9e4fb;border-radius:10px;background:#f9fbff;margin:8px 0}}
@media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <span class="pill">{html.escape(category)}</span>
    <h1>{html.escape(title)} Guided Builder</h1>
    <p class="muted"><strong>{html.escape(product.get("tagline", ""))}</strong></p>
    <p class="muted">{html.escape(product.get("description", ""))}</p>
  </section>

  <section class="grid">
    <aside class="side">
      <div class="step">1. Setup Profile</div>
      <div class="step">2. Customize Deliverables</div>
      <div class="step">3. Script Studio</div>
      <div class="step">4. Execution Board</div>
      <div class="step">5. Export Your Product</div>
      <div class="box"><strong>Included assets</strong><ul id="included-list" style="padding-left:18px;margin:8px 0 0"></ul></div>
    </aside>
    <main class="main">
      <div class="section">
        <h3>Setup Profile</h3>
        <label>Brand or Business Name</label>
        <input id="biz_name" placeholder="Your business name">
        <label>Target Audience</label>
        <input id="audience" placeholder="Who this is for">
        <label>Main Outcome</label>
        <input id="outcome" placeholder="What result buyer gets">
      </div>

      <div class="section">
        <h3>Customize Deliverables</h3>
        <div id="deliverables"></div>
      </div>

      <div class="section">
        <h3>Script Studio</h3>
        <div id="script_bank"></div>
        <label>Your Final Script</label>
        <textarea id="final_script" placeholder="Build your final customer-facing script here"></textarea>
      </div>

      <div class="section">
        <h3>Execution Board</h3>
        <table id="plan_tbl">
          <thead><tr><th>Step</th><th>Deliverable</th><th>Action</th><th>Due</th><th>Expected Outcome</th><th>Status</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>

      <div class="section">
        <h3>Export Your Product</h3>
        <div class="row">
          <button type="button" onclick="saveProgress()">Save Progress</button>
          <button type="button" class="alt" onclick="loadProgress()">Load Saved</button>
          <button type="button" onclick="downloadJson()">Download Product JSON</button>
          <button type="button" class="alt" onclick="downloadCsv()">Download Execution CSV</button>
        </div>
      </div>
    </main>
  </section>
</div>

<script>
const payload = {payload};
const includedList = document.getElementById("included-list");
payload.included.forEach(item => {{
  const li = document.createElement("li");
  li.textContent = item;
  includedList.appendChild(li);
}});

const deliverables = document.getElementById("deliverables");
payload.included.forEach((item, i) => {{
  const box = document.createElement("div");
  box.className = "box";
  box.innerHTML = `<label>${{item}} - Buyer-ready content</label><textarea id="deliverable_${{i}}" placeholder="Write the finalized content your buyer will actually use"></textarea>`;
  deliverables.appendChild(box);
}});

const bank = document.getElementById("script_bank");
payload.scripts.forEach(line => {{
  const b = document.createElement("button");
  b.type = "button";
  b.className = "alt";
  b.style.margin = "0 8px 8px 0";
  b.textContent = line;
  b.onclick = () => {{
    const t = document.getElementById("final_script");
    t.value = (t.value ? t.value + "\\n" : "") + line;
  }};
  bank.appendChild(b);
}});

const tbody = document.querySelector("#plan_tbl tbody");
payload.plan_rows.forEach((row, idx) => {{
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td>${{row[0] || `Step ${{idx+1}}`}}</td>
    <td contenteditable="true">${{row[1] || ""}}</td>
    <td contenteditable="true">${{row[2] || ""}}</td>
    <td contenteditable="true">${{row[3] || ""}}</td>
    <td contenteditable="true">${{row[4] || ""}}</td>
    <td>
      <select>
        <option>Not Started</option>
        <option>In Progress</option>
        <option>Completed</option>
      </select>
    </td>`;
  tbody.appendChild(tr);
}});

function captureState() {{
  return {{
    meta: {{
      title: payload.title,
      category: payload.category,
      biz_name: document.getElementById("biz_name").value,
      audience: document.getElementById("audience").value,
      outcome: document.getElementById("outcome").value
    }},
    deliverables: payload.included.map((_, i) => document.getElementById(`deliverable_${{i}}`)?.value || ""),
    final_script: document.getElementById("final_script").value,
    plan: [...document.querySelectorAll("#plan_tbl tbody tr")].map(r => {{
      const cells = [...r.querySelectorAll("td")];
      return {{
        step: cells[0]?.innerText?.trim() || "",
        deliverable: cells[1]?.innerText?.trim() || "",
        action: cells[2]?.innerText?.trim() || "",
        due: cells[3]?.innerText?.trim() || "",
        result: cells[4]?.innerText?.trim() || "",
        status: r.querySelector("select")?.value || "Not Started"
      }};
    }})
  }};
}}

function saveProgress() {{
  localStorage.setItem("northstar_guided_experience", JSON.stringify(captureState()));
  alert("Saved.");
}}

function loadProgress() {{
  const raw = localStorage.getItem("northstar_guided_experience");
  if (!raw) return;
  const state = JSON.parse(raw);
  document.getElementById("biz_name").value = state.meta?.biz_name || "";
  document.getElementById("audience").value = state.meta?.audience || "";
  document.getElementById("outcome").value = state.meta?.outcome || "";
  (state.deliverables || []).forEach((v, i) => {{
    const el = document.getElementById(`deliverable_${{i}}`);
    if (el) el.value = v || "";
  }});
  document.getElementById("final_script").value = state.final_script || "";
  if (state.plan?.length) {{
    const rows = [...document.querySelectorAll("#plan_tbl tbody tr")];
    rows.forEach((r, i) => {{
      const row = state.plan[i];
      if (!row) return;
      const cells = [...r.querySelectorAll("td")];
      if (cells[1]) cells[1].innerText = row.deliverable || "";
      if (cells[2]) cells[2].innerText = row.action || "";
      if (cells[3]) cells[3].innerText = row.due || "";
      if (cells[4]) cells[4].innerText = row.result || "";
      const sel = r.querySelector("select");
      if (sel) sel.value = row.status || "Not Started";
    }});
  }}
}}

function download(name, type, content) {{
  const blob = new Blob([content], {{type}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}}

function downloadJson() {{
  download("guided-experience-export.json", "application/json", JSON.stringify(captureState(), null, 2));
}}

function downloadCsv() {{
  const rows = [["Step","Deliverable","Action","Due","Expected Outcome","Status"]];
  captureState().plan.forEach(r => rows.push([r.step,r.deliverable,r.action,r.due,r.result,r.status]));
  const esc = v => `"${{String(v||"").replaceAll('"','""')}}"`;
  download("guided-execution-board.csv", "text/csv;charset=utf-8", rows.map(r => r.map(esc).join(",")).join("\\n"));
}}
</script>
</body>
</html>"""


def _build_retail_pack(product: dict, profile: dict) -> dict:
    title = product.get("title", "Digital Product")
    category = product.get("category", "General")
    preview = product.get("real_world_preview", {})
    included = product.get("preview_items", [])
    blueprint = _category_asset_blueprint(category)
    preview_cols = [str(c) for c in (preview.get("columns") or [])]
    preview_rows = [[str(c) for c in r] for r in (preview.get("rows") or [])]

    if title in WAVE1_PRODUCT_ASSETS:
        return WAVE1_PRODUCT_ASSETS[title]

    assets: list[dict] = []
    for idx, item in enumerate(included[:3], start=1):
        if idx == 1 and preview_cols and preview_rows:
            headers = preview_cols + ["Status"] if "Status" not in preview_cols else preview_cols[:]
            template_rows = []
            for row in preview_rows[:3]:
                row_vals = row[: len(preview_cols)]
                if "Status" not in preview_cols:
                    row_vals = row_vals + ["Planned"]
                template_rows.append(row_vals)
            completed_rows = []
            for row in template_rows:
                row_vals = row[:]
                if row_vals:
                    row_vals[-1] = "Completed"
                completed_rows.append(row_vals)
            assets.append(
                {
                    "name": item,
                    "headers": headers,
                    "template_rows": template_rows,
                    "completed_rows": completed_rows,
                    "purpose": f"Use this as the live working version of {item.lower()} for real customer outcomes.",
                }
            )
            continue

        core = blueprint["core_rows"][(idx - 1) % len(blueprint["core_rows"])] if blueprint["core_rows"] else []
        headers = blueprint["core_headers"][:]
        base = [str(v) for v in core] if core else [item, "Primary use case", "Customize for your workflow", "Today", "Buyer-ready output", "Ready"]
        while len(base) < len(headers):
            base.append("")
        template_rows = [base]
        completed = base[:]
        if completed:
            completed[-1] = "Completed"
        completed_rows = [completed]
        assets.append(
            {
                "name": item,
                "headers": headers,
                "template_rows": template_rows,
                "completed_rows": completed_rows,
                "purpose": f"Turn {item.lower()} into a finished, customer-ready deliverable.",
            }
        )

    while len(assets) < 3:
        fallback_idx = len(assets) + 1
        core = blueprint["core_rows"][(fallback_idx - 1) % len(blueprint["core_rows"])] if blueprint["core_rows"] else []
        headers = blueprint["core_headers"][:]
        base = [str(v) for v in core] if core else [f"Core Asset {fallback_idx}", "Primary use case", "Customize", "Today", "Ready output", "Ready"]
        while len(base) < len(headers):
            base.append("")
        completed = base[:]
        completed[-1] = "Completed"
        assets.append(
            {
                "name": f"Core Asset {fallback_idx}",
                "headers": headers,
                "template_rows": [base],
                "completed_rows": [completed],
                "purpose": "Finalize this core asset and deploy it in your workflow this week.",
            }
        )

    quickstart = [
        f"Open asset_01 and customize it for your exact use case.",
        f"Complete one live pass using {assets[0]['name']}.",
        f"Finalize {assets[1]['name']} and {assets[2]['name']} with your business details.",
        "Run weekly review and improve one bottleneck each cycle.",
    ]

    scripts = [
        f"Implementation line: this {category.lower()} toolkit is built for immediate practical use.",
        "Clarity line: keep instructions simple, outcome-first, and buyer-facing.",
        "Execution line: complete one full implementation pass before optimizing details.",
        "Result line: document measured outcomes and use them in your next iteration.",
    ]

    return {
        "buyer_result": preview.get("result") or profile.get("buyer_goal", "Practical implementation in real workflow."),
        "quickstart_steps": quickstart,
        "scripts": scripts,
        "assets": assets,
    }


def _customer_pack_files(product: dict) -> list[tuple[str, str]]:
    preview = product.get("real_world_preview", {})
    title = product.get("title", "Digital Product")
    category = product.get("category", "General")
    included = product.get("preview_items", [])
    price = f"${product.get('price_cents', 0) / 100:.2f}"
    safe_title = html.escape(title)
    safe_category = html.escape(category)
    safe_tagline = html.escape(product.get("tagline", ""))
    safe_description = html.escape(product.get("description", ""))
    safe_snippet = html.escape(product.get("preview_snippet", ""))
    profile = _product_content_profile(product)
    wave_pack = _build_retail_pack(product, profile)

    style = """
<style>
body{margin:0;background:#f4f7ff;color:#0f172a;font-family:Manrope,Segoe UI,sans-serif}
.wrap{max-width:940px;margin:0 auto;padding:28px 18px}
.card{background:#fff;border:1px solid #dbe3f0;border-radius:16px;padding:22px;box-shadow:0 10px 28px rgba(15,23,42,.08)}
h1,h2,h3{font-family:Sora,Manrope,sans-serif;letter-spacing:-.02em;margin:0 0 10px}
p{line-height:1.6;margin:0 0 10px;color:#334155}
.meta{display:inline-block;padding:6px 12px;border-radius:999px;background:#edf3ff;border:1px solid #d0defe;color:#1d4fb0;font-weight:700;font-size:13px;margin-bottom:10px}
ul{margin:0;padding-left:18px}
li{margin:6px 0;line-height:1.5}
table{width:100%;border-collapse:collapse;font-size:14px;background:#fff}
th,td{padding:10px;border:1px solid #e4ecfb;text-align:left;vertical-align:top}
th{background:#f7faff;color:#1e3a5f}
.cta{margin-top:14px;padding:14px;border:1px solid #d4e3ff;background:#f2f7ff;border-radius:12px;color:#1e3a8a;font-weight:700}
</style>
"""

    included_html = "".join(f"<li>{html.escape(item)}</li>" for item in included)
    plan_rows_html = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        + "</tr>"
        for row in profile["plan_rows"]
    )
    step_list = wave_pack.get("quickstart_steps") or profile["quickstart_steps"]
    script_bank = wave_pack.get("scripts") or profile["scripts"]
    step_list_html = "".join(f"<li>{html.escape(step)}</li>" for step in step_list)
    script_lines = [f"- {line}" for line in script_bank]

    def _asset_doc(item: str, idx: int) -> str:
        safe_item = html.escape(item)
        asset_specs = wave_pack.get("assets") or []
        spec = asset_specs[idx] if idx < len(asset_specs) else {}
        asset_name = spec.get("name") or item
        purpose = spec.get("purpose")
        table_headers = spec.get("headers") or blueprint["core_headers"][:]
        template_rows = spec.get("template_rows") or []
        completed_rows = spec.get("completed_rows") or []

        workflow = blueprint["core_rows"][idx % len(blueprint["core_rows"])] if blueprint["core_rows"] else []
        if not template_rows:
            base_row = [str(c) for c in workflow] if workflow else [asset_name, "Primary use case", f"Customize {item}", "Today", "Ready outcome", "Ready"]
            while len(base_row) < len(table_headers):
                base_row.append("")
            template_rows = [base_row]
        if not completed_rows:
            completed_rows = [list(r) for r in template_rows]
            for row in completed_rows:
                if row:
                    row[-1] = "Completed"

        quick_action = template_rows[0][2] if template_rows and len(template_rows[0]) > 2 else f"Customize {asset_name} for your use case"
        expected_outcome = (
            completed_rows[0][4]
            if completed_rows and len(completed_rows[0]) > 4 and completed_rows[0][4]
            else f"{asset_name} ready for immediate buyer use"
        )
        header_html = "".join(f"<th>{html.escape(str(h))}</th>" for h in table_headers)
        template_html_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in template_rows
        )
        example_html_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in completed_rows
        )

        use_case = {
            "Creator Growth": "Use this to plan and publish high-intent content each week.",
            "Creator Business": "Use this to package and pitch a paid offer clearly.",
            "Ecommerce": "Use this to improve listing quality and conversion.",
            "Finance": "Use this to make weekly money decisions with confidence.",
            "Career": "Use this to prepare measurable job materials and interviews.",
            "Freelance Ops": "Use this to reduce scope creep and speed up delivery.",
            "Wellness": "Use this to run a realistic weekly plan and track adherence.",
            "Family": "Use this to coordinate routines with less daily friction.",
            "Events": "Use this to deliver polished event assets on schedule.",
            "Hospitality": "Use this to improve guest communication and consistency.",
            "Operations": "Use this to standardize recurring workflows and handoffs.",
            "Productivity": "Use this to execute priorities without context switching.",
            "Branding": "Use this to keep brand voice and visuals consistent.",
            "Real Estate": "Use this to move leads through a reliable follow-up sequence.",
        }.get(category, "Use this to implement the product in a real workflow.")

        return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{style}</head><body>
<div class="wrap"><section class="card">
<div class="meta">{safe_category}</div>
<h1>{html.escape(asset_name)}</h1>
<p><strong>Purpose:</strong> {html.escape(purpose or use_case)}</p>
<p><strong>15-minute setup:</strong> {html.escape(quick_action)}.</p>
<h3>Template you can fill now</h3>
<table><thead><tr>{header_html}</tr></thead><tbody>{template_html_rows}</tbody></table>
<h3>Completed example</h3>
<table><thead><tr>{header_html}</tr></thead><tbody>{example_html_rows}</tbody></table>
<div class="cta">Expected outcome: {html.escape(expected_outcome)}</div>
<p>Next: copy this structure into your working tool, replace sample fields, and run one real-world pass today.</p>
</section></div></body></html>"""

    readme_txt = "\n".join(
        [
            f"{title} - Buyer Delivery Pack",
            "",
            "START HERE",
            "1) Open 01_Start_Here.html",
            "2) Open 02_Quickstart_Guide.html for the guided onboarding path",
            "3) Open each asset_XX HTML file and customize it for your real use case",
            "4) Use CSV files as optional import/export support files",
            "5) Use 04_Copy_Paste_Scripts.txt and 05_Performance_Tracker.csv weekly",
            "",
            "This package is built for immediate implementation and customer-facing delivery quality.",
        ]
    )

    start_here_html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{style}</head><body>
<div class="wrap"><section class="card">
<div class="meta">{safe_category}</div>
<h1>{safe_title}</h1>
<p><strong>{safe_tagline}</strong></p>
<p>{safe_description}</p>
<h3>What you get</h3>
<ul>{included_html}</ul>
<h3>Fast Start Checklist</h3>
<ul>{step_list_html}</ul>
<div class="cta">Quick win: {html.escape(profile["quick_win"])}</div>
<p><strong>Buyer goal:</strong> {html.escape(profile["buyer_goal"])}</p>
<p><strong>Price paid:</strong> {price}</p>
<p><strong>Important:</strong> The <code>asset_XX_*.html</code> files are your primary buyer-ready product assets.</p>
</section></div></body></html>"""

    quickstart_guide_html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{style}</head><body>
<div class="wrap"><section class="card">
<h1>Quickstart Guide</h1>
<p>Use this sequence to get from download to first real result quickly.</p>
<ol>
  <li>Open and customize each <strong>asset_XX HTML</strong> file in order.</li>
  <li>Complete the first pass in one working session.</li>
  <li>Use the script bank for communication or publishing copy.</li>
  <li>Track outcomes weekly and improve one bottleneck at a time.</li>
</ol>
<h3>Fast Start Checklist</h3>
<ul>{step_list_html}</ul>
<p><strong>Target result:</strong> {html.escape(wave_pack.get("buyer_result", profile["buyer_goal"]))}</p>
<div class="cta">Most buyers get first implementation done in under 90 minutes.</div>
</section></div></body></html>"""

    workbook_html_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in profile["workbook_rows"]
    )
    workbook_html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{style}</head><body>
<div class="wrap"><section class="card">
<h1>Master Workbook</h1>
<p>This is your execution board for setup, launch, and optimization.</p>
<table>
<thead><tr><th>ID</th><th>Deliverable</th><th>Action</th><th>Due</th><th>Status</th><th>Definition Of Done</th><th>Notes</th></tr></thead>
<tbody>{workbook_html_rows}</tbody></table>
<div class="cta">Rule: do not move to optimization until your first full implementation pass is complete.</div>
</section></div></body></html>"""

    playbook_html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{style}</head><body>
<div class="wrap"><section class="card">
<div class="meta">{safe_category}</div>
<h1>{safe_title} - Implementation Playbook</h1>
<p>{safe_snippet}</p>
<h3>Execution plan</h3>
<table>
<thead><tr><th>Step</th><th>Deliverable</th><th>Action</th><th>Timeline</th><th>Expected outcome</th></tr></thead>
<tbody>{plan_rows_html}</tbody>
</table>
<h3>Pro scripts</h3>
<ul>{"".join(f"<li>{html.escape(line)}</li>" for line in script_bank)}</ul>
<div class="cta">{html.escape(preview.get("result", "Outcome: implement this pack in your workflow and track performance weekly."))}</div>
</section></div></body></html>"""

    setup_html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{style}</head><body>
<div class="wrap"><section class="card">
<h1>90-Minute Setup Sprint</h1>
<p>Block 90 minutes and execute this setup flow in order.</p>
<table>
<thead><tr><th>Time Block</th><th>Task</th><th>Output</th></tr></thead>
<tbody>
<tr><td>0-20 min</td><td>Customize core template fields</td><td>First draft completed</td></tr>
<tr><td>20-45 min</td><td>Apply scripts/checklists to your context</td><td>Ready-to-use workflow</td></tr>
<tr><td>45-70 min</td><td>Run one real-world use case</td><td>Validated output</td></tr>
<tr><td>70-90 min</td><td>Set KPIs + 30-day cadence</td><td>Tracking + accountability</td></tr>
</tbody>
</table>
<div class="cta">Do not optimize before your first full implementation pass.</div>
</section></div></body></html>"""

    license_html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">""" + style + """</head><body>
<div class="wrap"><section class="card">
<h1>License + 7-Day Satisfaction Guarantee</h1>
<p>License: one buyer, internal business/household use, no redistribution or resale.</p>
<p>Satisfaction guarantee: one store-credit request per payment within 7 days of purchase, redeemable toward another Northstar Studio product.</p>
</section></div></body></html>"""

    quickstart_csv = _csv_block(
        ["Task ID", "Deliverable", "Action", "Due", "Status", "Output", "Notes"],
        [[f"Q{idx+1:02d}", row[1], row[2], row[3], "Not Started", row[4], ""] for idx, row in enumerate(profile["plan_rows"][:5])],
    )
    workbook_csv = _csv_block(
        ["Workstream ID", "Deliverable", "Action", "Due", "Status", "Definition of done", "Notes"],
        profile["workbook_rows"],
    )
    tracker_csv = _csv_block(
        ["Metric", "Baseline", "Target", "Week 1", "Week 2", "Week 3", "Week 4", "Notes"],
        profile["kpi_rows"],
    )
    calendar_csv = _csv_block(
        ["Day", "Primary Task", "Expected Result", "Status", "Notes"],
        profile["calendar_rows"],
    )

    sample_rows = preview.get("rows") or [row[1:] for row in profile["plan_rows"][:4]]
    sample_headers = preview.get("columns") or ["Deliverable", "Action", "Expected Result"]
    sample_csv_rows = [[str(c) for c in row[: len(sample_headers)]] for row in sample_rows]
    sample_csv = _csv_block(sample_headers, sample_csv_rows)

    scripts_txt = "\n".join(
        [
            f"{title} - Copy/Paste Script Library",
            "",
            "Use these scripts directly and customize bracketed fields:",
            "",
            *script_lines,
            "",
            "Customer message template:",
            "- Thanks for your purchase. Start with 01_Start_Here.html, then customize each asset_XX HTML file.",
            "- Complete one finalized asset today and use it in your real workflow.",
        ]
    )

    workspace_rows = "".join(
        f"<tr><td>{html.escape(row[0])}</td><td contenteditable='true'>{html.escape(row[2])}</td>"
        f"<td contenteditable='true'>{html.escape(row[3])}</td><td contenteditable='true'>{html.escape(row[4])}</td></tr>"
        for row in profile["plan_rows"]
    )
    workspace_html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{style}</head><body>
<div class="wrap"><section class="card"><div class="meta">{safe_category}</div><h1>{safe_title} Interactive Builder</h1>
<p>Edit any cell, add rows, and export your customized plan as CSV.</p>
<table id="planner"><thead><tr><th>Step</th><th>Action</th><th>Due</th><th>Expected Result</th></tr></thead><tbody>{workspace_rows}</tbody></table>
<div style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap">
<button onclick="addRow()">Add Row</button>
<button onclick="savePlan()">Save Local</button>
<button onclick="loadPlan()">Load Saved</button>
<button onclick="exportCsv()">Export CSV</button>
</div>
</section></div>
<script>
function toRows() {{
  return [...document.querySelectorAll("#planner tbody tr")].map(r => [...r.children].map(c => c.innerText.trim()));
}}
function addRow() {{
  const tr=document.createElement("tr");
  tr.innerHTML="<td contenteditable='true'>New Step</td><td contenteditable='true'>Add action</td><td contenteditable='true'>This week</td><td contenteditable='true'>Expected result</td>";
  document.querySelector("#planner tbody").appendChild(tr);
}}
function savePlan() {{
  localStorage.setItem("northstar-plan", JSON.stringify(toRows()));
  alert("Saved in this browser.");
}}
function loadPlan() {{
  const raw=localStorage.getItem("northstar-plan");
  if(!raw) return;
  const rows=JSON.parse(raw);
  const body=document.querySelector("#planner tbody");
  body.innerHTML="";
  rows.forEach(row => {{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td contenteditable='true'>${{row[0]||""}}</td><td contenteditable='true'>${{row[1]||""}}</td><td contenteditable='true'>${{row[2]||""}}</td><td contenteditable='true'>${{row[3]||""}}</td>`;
    body.appendChild(tr);
  }});
}}
function exportCsv() {{
  const headers=["Step","Action","Due","Expected Result"];
  const rows=[headers,...toRows()];
  const csv=rows.map(r=>r.map(v=>`"${{String(v).replaceAll('"','""')}}"`).join(",")).join("\\n");
  const blob=new Blob([csv],{{type:"text/csv;charset=utf-8;"}});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="interactive-builder.csv"; a.click();
}}
</script></body></html>"""

    blueprint = _category_asset_blueprint(category)
    core_template_csv = _csv_block(blueprint["core_headers"], blueprint["core_rows"])
    filled_rows = [list(r) for r in blueprint["core_rows"]]
    if filled_rows:
        filled_rows[0][-1] = "Completed"
    filled_example_csv = _csv_block(blueprint["core_headers"], filled_rows)
    quality_checklist_txt = "\n".join(
        [
            f"{title} - Quality Checklist",
            "",
            "Before delivering or publishing, confirm:",
            "- The buyer-facing file has real, customized content.",
            "- No placeholder/internal notes remain.",
            "- Dates, names, and numbers are accurate.",
            "- The file can be used immediately by a customer.",
            "- A completed example is included.",
        ]
    )
    script_bank_txt = "\n".join([f"{title} - Script Bank", ""] + [f"- {line}" for line in blueprint["script_bank"]])
    if wave_pack.get("scripts"):
        script_bank_txt = "\n".join([f"{title} - Script Bank", ""] + [f"- {line}" for line in wave_pack["scripts"]])

    files = [
        ("00_READ_FIRST.txt", readme_txt),
        ("01_Start_Here.html", start_here_html),
        ("02_Quickstart_Guide.html", quickstart_guide_html),
        ("02_Quickstart_Action_Plan.csv", quickstart_csv),
        ("03_Master_Workbook.html", workbook_html),
        ("03_Master_Workbook.csv", workbook_csv),
        ("04_Copy_Paste_Scripts.txt", scripts_txt),
        ("05_Performance_Tracker.csv", tracker_csv),
        ("06_30_Day_Execution_Calendar.csv", calendar_csv),
        ("07_Implementation_Playbook.html", playbook_html),
        ("08_Filled_Example.csv", sample_csv),
        ("09_90_Minute_Setup_Sprint.html", setup_html),
        ("10_Interactive_Builder.html", workspace_html),
        ("11_License_and_Guarantee.html", license_html),
        ("12_Core_Product_Template.csv", core_template_csv),
        ("13_Core_Product_Completed_Example.csv", filled_example_csv),
        ("14_Script_Bank.txt", script_bank_txt),
        ("15_Quality_Checklist.txt", quality_checklist_txt),
        ("16_Guided_Interactive_Experience.html", _guided_experience_html(product)),
    ]
    for i, item in enumerate(included, start=1):
        asset_specs = wave_pack.get("assets") or []
        asset_name = asset_specs[i - 1].get("name") if i - 1 < len(asset_specs) else item
        item_sheet_csv = _csv_block(
            ["Section", "Buyer Action", "Filled Example"],
            [
                [f"{asset_name} - Setup", "Customize with your real details", f"{asset_name} customized and ready"],
                [f"{asset_name} - Execution", "Use in your live workflow", f"{asset_name} used with first customer/run"],
                [f"{asset_name} - Review", "Capture improvements", f"{asset_name} optimized for next cycle"],
            ],
        )
        files.append((f"asset_{i:02d}_{slugify(asset_name)[:42]}.html", _asset_doc(item, i - 1)))
        files.append((f"asset_{i:02d}_{slugify(asset_name)[:42]}.csv", item_sheet_csv))
    return files


def build_customer_product_pack(product: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in _customer_pack_files(product):
            zf.writestr(name, content)
    return buf.getvalue()


def build_customer_bundle_pack(bundle: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "00_Bundle_Start_Here.html",
            f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:Manrope,Segoe UI,sans-serif;background:#f4f7ff;color:#0f172a}}.wrap{{max-width:820px;margin:30px auto;padding:0 18px}}
.card{{background:#fff;border:1px solid #dbe3f0;border-radius:16px;padding:22px}}h1{{font-family:Sora,Manrope,sans-serif}}</style></head>
<body><div class="wrap"><section class="card"><h1>{html.escape(bundle['title'])}</h1>
<p>Thank you for purchasing this Northstar Studio bundle.</p>
<p>Open each product folder and start with <strong>01_Start_Here.html</strong>.</p></section></div></body></html>""",
        )
        for item in bundle.get("items", []):
            product = get_product_by_title(item["title"])
            if not product:
                continue
            folder = f"{slugify(product['title'])}/"
            for name, content in _customer_pack_files(product):
                zf.writestr(folder + name, content)
    return buf.getvalue()


def build_product_qc_zip(product: dict) -> bytes:
    title = product.get("title", "Untitled Product")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        qc_meta = {
            "title": title,
            "category": product.get("category", "General"),
            "price_cents": product.get("price_cents", 0),
            "tagline": product.get("tagline", ""),
            "description": product.get("description", ""),
            "included": product.get("preview_items", []),
            "preview": product.get("real_world_preview", {}),
        }
        zf.writestr("00_qc_metadata.json", json.dumps(qc_meta, indent=2))
        for file_name, content in _customer_pack_files(product):
            zf.writestr(file_name, content)
    return buf.getvalue()


def _word_set(text: str) -> set[str]:
    tokens: list[str] = []
    current = []
    for ch in (text or "").lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return {t for t in tokens if len(t) > 2}


def _occupation_focus(occupation: str) -> dict:
    o = (occupation or "").lower()
    mapping = {
        "fitness": {"goal": "deliver client training plans", "keywords": ["workout", "habit", "plan", "coach", "health"]},
        "coach": {"goal": "deliver premium coaching assets", "keywords": ["client", "program", "offer", "onboarding", "delivery"]},
        "teacher": {"goal": "organize curriculum and family routines", "keywords": ["planner", "schedule", "family", "worksheet", "routine"]},
        "freelancer": {"goal": "run client ops and proposals", "keywords": ["scope", "proposal", "invoice", "client", "retainer"]},
        "realtor": {"goal": "convert and nurture leads", "keywords": ["lead", "buyer", "listing", "followup", "consult"]},
        "airbnb": {"goal": "improve guest experience", "keywords": ["guest", "welcome", "host", "checkout", "message"]},
        "creator": {"goal": "publish and monetize content", "keywords": ["content", "hook", "caption", "audience", "cta"]},
        "ecommerce": {"goal": "improve listing conversion", "keywords": ["listing", "product", "keyword", "seo", "shop"]},
        "finance": {"goal": "track cash flow and planning", "keywords": ["budget", "cashflow", "tracker", "revenue", "savings"]},
        "operations": {"goal": "standardize team workflows", "keywords": ["sop", "process", "checklist", "owner", "handoff"]},
    }
    for k, v in mapping.items():
        if k in o:
            return v
    return {"goal": "use ready-to-run professional templates", "keywords": _word_set(occupation) or {"template", "planner", "checklist"}}


def run_virtual_product_test(product: dict, occupation: str) -> dict:
    focus = _occupation_focus(occupation)
    expected_assets = len(product.get("preview_items", []))
    zip_bytes = build_customer_product_pack(product)

    required_files = {
        "00_READ_FIRST.txt",
        "01_Start_Here.html",
        "02_Quickstart_Guide.html",
        "02_Quickstart_Action_Plan.csv",
        "03_Master_Workbook.html",
        "03_Master_Workbook.csv",
        "07_Implementation_Playbook.html",
        "10_Interactive_Builder.html",
        "11_License_and_Guarantee.html",
        "16_Guided_Interactive_Experience.html",
    }
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode="r") as zf:
        names = set(zf.namelist())
        missing = sorted(required_files - names)
        asset_stems = sorted(
            {
                n.rsplit(".", 1)[0]
                for n in names
                if n.startswith("asset_") and (n.endswith(".csv") or n.endswith(".html"))
            }
        )
        asset_files = sorted([n for n in names if n.startswith("asset_")])[:20]
        sample_names = sorted(list(names))[:20]

        text_blob_parts = []
        for name in zf.namelist():
            if not name.endswith((".txt", ".csv", ".html", ".md", ".json")):
                continue
            try:
                text_blob_parts.append(zf.read(name).decode("utf-8", errors="ignore"))
            except Exception:
                continue
        text_blob = "\n".join(text_blob_parts)

    bad_markers = ["lorem ipsum", "[insert", "todo", "tbd", "dummy text", "placeholder"]
    marker_hits = [m for m in bad_markers if m in text_blob.lower()]

    product_text = " ".join(
        [
            product.get("title", ""),
            product.get("category", ""),
            product.get("tagline", ""),
            product.get("description", ""),
            " ".join(product.get("preview_items", [])),
            text_blob[:12000],
        ]
    )
    product_words = _word_set(product_text)
    focus_words = set(focus["keywords"])
    overlap = sorted(list(product_words & focus_words))

    completeness = max(0, 100 - len(missing) * 12 - max(0, expected_assets - len(asset_stems)) * 8)
    usability = 100
    if "16_Guided_Interactive_Experience.html" not in names:
        usability -= 30
    if "10_Interactive_Builder.html" not in names:
        usability -= 20
    if len(asset_stems) == 0:
        usability -= 30
    clarity = max(0, 100 - len(marker_hits) * 20)
    relevance = min(100, 35 + len(overlap) * 12)

    overall = int(round((completeness * 0.30) + (usability * 0.30) + (clarity * 0.20) + (relevance * 0.20)))
    passed = overall >= 80 and not missing and len(marker_hits) == 0

    occupation_goal = focus["goal"]
    usage_simulation = (
        f"Simulated buyer ({occupation}): downloads pack, opens guided experience, "
        f"customizes first asset, and uses it to {occupation_goal}."
    )

    issues = []
    if missing:
        issues.append(f"Missing required files: {', '.join(missing)}")
    if len(asset_stems) < expected_assets:
        issues.append(f"Asset file count is low ({len(asset_stems)}/{expected_assets} expected from preview items).")
    if marker_hits:
        issues.append(f"Placeholder markers detected: {', '.join(marker_hits)}")
    if relevance < 60:
        issues.append("Occupation relevance is weak; deliverables should be tuned more directly to this buyer type.")

    recommendations = []
    if missing:
        recommendations.append("Regenerate the pack and ensure all required retail files are included.")
    if len(asset_stems) < expected_assets:
        recommendations.append("Create one concrete asset file per listed deliverable.")
    if marker_hits:
        recommendations.append("Remove placeholder language and replace with finalized, buyer-usable copy.")
    if relevance < 60:
        recommendations.append("Align examples/scripts to the target occupation's daily workflow and outcomes.")
    if not recommendations:
        recommendations.append("Pack is sale-ready for this occupation test. Continue with human QA spot checks.")

    return {
        "ok": True,
        "passed": passed,
        "overall_score": overall,
        "scores": {
            "completeness": int(completeness),
            "usability": int(usability),
            "clarity": int(clarity),
            "occupation_relevance": int(relevance),
        },
        "occupation": occupation,
        "occupation_goal": occupation_goal,
        "product": {
            "id": product.get("id"),
            "title": product.get("title"),
            "category": product.get("category"),
            "price_cents": product.get("price_cents"),
        },
        "download_validation": {
            "required_files_missing": missing,
            "asset_files_found": asset_files,
            "asset_units_found": asset_stems,
            "expected_asset_count": expected_assets,
            "sample_files_checked": sample_names,
        },
        "simulated_usage": usage_simulation,
        "relevance_overlap_keywords": overlap,
        "issues": issues,
        "recommendations": recommendations,
    }


def _sales_funnel_snapshot(days: int = 30) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=max(1, days))).isoformat()
    conn = db()
    sales_row = conn.execute(
        "SELECT COUNT(*) AS c, COALESCE(SUM(amount_cents), 0) AS gross FROM sales WHERE created_at >= ?",
        (since,),
    ).fetchone()
    leads_row = conn.execute(
        "SELECT COUNT(*) AS c FROM leads WHERE created_at >= ?",
        (since,),
    ).fetchone()
    deliveries_row = conn.execute(
        "SELECT COUNT(*) AS c, SUM(CASE WHEN status='sent' THEN 1 ELSE 0 END) AS sent FROM deliveries WHERE created_at >= ?",
        (since,),
    ).fetchone()
    conn.close()
    sales_count = int(sales_row["c"] if sales_row else 0)
    leads_count = int(leads_row["c"] if leads_row else 0)
    delivery_total = int(deliveries_row["c"] if deliveries_row else 0)
    delivery_sent = int(deliveries_row["sent"] if deliveries_row and deliveries_row["sent"] is not None else 0)
    gross = int(sales_row["gross"] if sales_row else 0)
    lead_to_sale = round((sales_count / leads_count) * 100, 1) if leads_count else 0.0
    delivery_rate = round((delivery_sent / delivery_total) * 100, 1) if delivery_total else 0.0
    return {
        "window_days": days,
        "sales_count": sales_count,
        "gross_cents": gross,
        "gross_usd": round(gross / 100, 2),
        "leads_count": leads_count,
        "lead_to_sale_rate_pct": lead_to_sale,
        "delivery_events": delivery_total,
        "delivery_success_rate_pct": delivery_rate,
    }


def _category_distribution(products: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for p in products:
        cat = p.get("category", "General")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _pricing_analysis(products: list[dict]) -> list[dict]:
    by_category: dict[str, list[int]] = {}
    for p in products:
        by_category.setdefault(p.get("category", "General"), []).append(int(p.get("price_cents", 0)))
    results: list[dict] = []
    for cat, vals in by_category.items():
        avg = int(sum(vals) / max(1, len(vals)))
        lo = min(vals)
        hi = max(vals)
        spread_pct = round(((hi - lo) / max(1, avg)) * 100, 1)
        results.append(
            {
                "category": cat,
                "count": len(vals),
                "avg_price_usd": round(avg / 100, 2),
                "min_price_usd": round(lo / 100, 2),
                "max_price_usd": round(hi / 100, 2),
                "spread_pct_of_avg": spread_pct,
            }
        )
    return sorted(results, key=lambda x: x["count"], reverse=True)


def build_ai_team_report(occupation: str, limit: int = 50) -> dict:
    products = list_products()[: max(1, min(200, limit))]
    qa_reports = [run_virtual_product_test(p, occupation) for p in products]
    failed = [r for r in qa_reports if not r.get("passed")]
    failed_sorted = sorted(failed, key=lambda r: int(r.get("overall_score", 0)))
    avg_score = int(round(sum(int(r.get("overall_score", 0)) for r in qa_reports) / max(1, len(qa_reports))))

    category_counts = _category_distribution(products)
    thin_categories = [{"category": c, "count": n} for c, n in category_counts.items() if n < 3]
    thin_categories = sorted(thin_categories, key=lambda x: x["count"])

    pricing = _pricing_analysis(products)
    funnel = _sales_funnel_snapshot(days=30)

    priorities: list[dict] = []
    if failed_sorted:
        top = failed_sorted[:10]
        priorities.append(
            {
                "owner": "Product QA Agent",
                "priority": "P0",
                "why": f"{len(failed_sorted)} products are below sale-ready threshold.",
                "action": "Fix the 10 lowest-scoring products first using issues from the QA report.",
                "targets": [{"title": r["product"]["title"], "score": r["overall_score"]} for r in top],
            }
        )
    if thin_categories:
        priorities.append(
            {
                "owner": "Market Agent",
                "priority": "P1",
                "why": "Category coverage is thin in some areas.",
                "action": "Add at least 2 high-quality products in each thin category.",
                "targets": thin_categories,
            }
        )
    if funnel["lead_to_sale_rate_pct"] < 5:
        priorities.append(
            {
                "owner": "Conversion Agent",
                "priority": "P1",
                "why": f"Lead-to-sale rate is low at {funnel['lead_to_sale_rate_pct']}%.",
                "action": "Improve product page proof, add stronger offer framing, and tighten checkout copy.",
                "targets": [{"metric": "lead_to_sale_rate_pct", "value": funnel["lead_to_sale_rate_pct"]}],
            }
        )
    if funnel["delivery_success_rate_pct"] < 95:
        priorities.append(
            {
                "owner": "Delivery Agent",
                "priority": "P1",
                "why": f"Delivery success is {funnel['delivery_success_rate_pct']}%.",
                "action": "Audit failed delivery notes and harden email delivery fallback.",
                "targets": [{"metric": "delivery_success_rate_pct", "value": funnel["delivery_success_rate_pct"]}],
            }
        )
    if not priorities:
        priorities.append(
            {
                "owner": "Growth Agent",
                "priority": "P2",
                "why": "Core product quality and funnel health are stable.",
                "action": "Scale top categories and increase average order value with curated bundles.",
                "targets": [],
            }
        )

    return {
        "ok": True,
        "generated_at": utc_now_iso(),
        "occupation_target": occupation,
        "products_scanned": len(products),
        "qa_summary": {
            "average_score": avg_score,
            "passed": len(qa_reports) - len(failed),
            "failed": len(failed),
            "threshold": 80,
        },
        "market_agent": {
            "category_distribution": category_counts,
            "thin_categories": thin_categories,
        },
        "pricing_agent": {
            "category_pricing": pricing,
        },
        "funnel_agent": funnel,
        "priority_actions": priorities,
        "lowest_scoring_products": [
            {
                "id": r["product"]["id"],
                "title": r["product"]["title"],
                "category": r["product"]["category"],
                "score": r["overall_score"],
                "issues": r.get("issues", []),
                "recommendations": r.get("recommendations", []),
            }
            for r in failed_sorted[:15]
        ],
    }


def _occupation_language_pack(occupation: str) -> dict:
    o = (occupation or "").strip()
    ol = o.lower()
    if "fitness" in ol or "coach" in ol:
        return {
            "audience": "fitness coaches and wellness professionals",
            "value_theme": "client results, adherence, and clear implementation",
            "benefit_line": "get buyer-ready tools that improve consistency and outcomes this week",
        }
    if "freelance" in ol:
        return {
            "audience": "freelancers and solo service operators",
            "value_theme": "scope clarity, smoother delivery, and faster payment cycles",
            "benefit_line": "ship client work with less back-and-forth and better margins",
        }
    if "realtor" in ol or "real estate" in ol:
        return {
            "audience": "real estate agents and lead-focused teams",
            "value_theme": "lead conversion, follow-up consistency, and trust",
            "benefit_line": "convert more leads into booked consults and closed deals",
        }
    if "creator" in ol:
        return {
            "audience": "content creators building monetized audiences",
            "value_theme": "production speed, conversion clarity, and repeatability",
            "benefit_line": "publish faster and turn attention into sales-ready demand",
        }
    return {
        "audience": f"{o} buyers" if o else "small business buyers",
        "value_theme": "immediate practical value and buyer confidence",
        "benefit_line": "use clear, professional templates to create real outcomes quickly",
    }


def _autofix_content_for_product(product: dict, occupation: str) -> dict:
    lang = _occupation_language_pack(occupation)
    title = product.get("title", "Digital Product")
    category = product.get("category", "General")
    base_items = product.get("preview_items", [])
    while len(base_items) < 3:
        base_items.append(f"{title} core template component {len(base_items) + 1}")
    preview_items = [base_items[0], base_items[1], base_items[2]]
    tagline = f"Practical toolkit for {lang['audience']} focused on {lang['value_theme']}."
    description = (
        f"A practical digital product for {lang['audience']} focused on {lang['value_theme']}. "
        f"Inside the package: editable core assets, completed examples, and a guided workflow to help buyers "
        f"{lang['benefit_line']} without guesswork."
    )
    preview_snippet = (
        f"How this is used in the real world: open the guided experience, customize each deliverable in order, "
        f"run one live implementation, then track weekly outcomes. The goal is to {lang['benefit_line']}."
    )
    return {
        "title": title,
        "category": category,
        "tagline": tagline,
        "description": description,
        "preview_items": preview_items,
        "preview_snippet": preview_snippet,
    }


def run_ai_team_autofix(occupation: str, limit: int = 50, min_score: int = 80, max_updates: int = 20) -> dict:
    report = build_ai_team_report(occupation, limit=limit)
    fail_candidates = report.get("lowest_scoring_products", [])
    to_fix = [x for x in fail_candidates if int(x.get("score", 0)) < min_score][: max(1, min(200, max_updates))]
    updates = []
    for item in to_fix:
        product = get_product(item.get("id", ""))
        if not product:
            continue
        fixed = _autofix_content_for_product(product, occupation)
        save_content_override(
            title=fixed["title"],
            category=fixed["category"],
            tagline=fixed["tagline"],
            description=fixed["description"],
            preview_items=fixed["preview_items"],
            preview_snippet=fixed["preview_snippet"],
        )
        updates.append(
            {
                "id": product["id"],
                "title": product["title"],
                "old_score": int(item.get("score", 0)),
                "new_tagline": fixed["tagline"],
            }
        )
    return {
        "ok": True,
        "occupation_target": occupation,
        "min_score_threshold": min_score,
        "max_updates": max_updates,
        "updated_count": len(updates),
        "updated_products": updates,
    }


def generate_ai_fix_proposals(occupation: str, limit: int = 50, min_score: int = 80, max_proposals: int = 20) -> dict:
    report = build_ai_team_report(occupation, limit=limit)
    fail_candidates = report.get("lowest_scoring_products", [])
    to_fix = [x for x in fail_candidates if int(x.get("score", 0)) < min_score][: max(1, min(200, max_proposals))]
    created = []
    for item in to_fix:
        product = get_product(item.get("id", ""))
        if not product:
            continue
        payload = _autofix_content_for_product(product, occupation)
        proposal = save_ai_fix_proposal(
            product_id=product["id"],
            title=product["title"],
            occupation=occupation,
            prior_score=int(item.get("score", 0)),
            payload=payload,
            status="pending",
            notes="ai generated proposal",
        )
        created.append(proposal)
    return {
        "ok": True,
        "occupation_target": occupation,
        "min_score_threshold": min_score,
        "created_count": len(created),
        "proposals": created,
    }


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
            "headline": "Implementation Planner",
            "subhead": "A practical worksheet buyers use to launch this product quickly.",
            "columns": ["Priority", "Action Item", "Owner", "Due Date", "Success Metric"],
            "rows": [
                ["High", "Define your first deliverable outcome", "You", "This week", "Clear outcome selected"],
                ["High", "Customize template with your business details", "You", "This week", "Draft completed"],
                ["Medium", "Run first implementation pass", "You", "Next 7 days", "First version shipped"],
                ["Medium", "Collect feedback and refine", "You", "Next 14 days", "Improved final version"],
                ["Low", "Archive reusable version for future use", "You", "This month", "Reusable system saved"],
            ],
            "result": "Outcome: a clean first implementation and repeatable workflow in under 30 minutes.",
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS product_content_overrides (
            title TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            tagline TEXT NOT NULL,
            description TEXT NOT NULL,
            preview_items_json TEXT NOT NULL,
            preview_snippet TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_fix_proposals (
            id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            title TEXT NOT NULL,
            occupation TEXT NOT NULL,
            prior_score INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            notes TEXT,
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


def get_content_override(title: str) -> dict | None:
    conn = db()
    row = conn.execute(
        """
        SELECT title, category, tagline, description, preview_items_json, preview_snippet, updated_at
        FROM product_content_overrides
        WHERE title = ?
        """,
        (title,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    try:
        data["preview_items"] = json.loads(data.get("preview_items_json") or "[]")
    except Exception:
        data["preview_items"] = []
    return data


def save_content_override(
    title: str,
    category: str,
    tagline: str,
    description: str,
    preview_items: list[str],
    preview_snippet: str,
) -> None:
    conn = db()
    conn.execute(
        """
        INSERT INTO product_content_overrides
        (title, category, tagline, description, preview_items_json, preview_snippet, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(title) DO UPDATE SET
            category=excluded.category,
            tagline=excluded.tagline,
            description=excluded.description,
            preview_items_json=excluded.preview_items_json,
            preview_snippet=excluded.preview_snippet,
            updated_at=excluded.updated_at
        """,
        (
            title,
            category,
            tagline,
            description,
            json.dumps(preview_items),
            preview_snippet,
            utc_now_iso(),
        ),
    )
    conn.commit()
    conn.close()


def list_content_overrides(limit: int = 200) -> list[dict]:
    conn = db()
    rows = conn.execute(
        """
        SELECT title, category, tagline, preview_snippet, updated_at
        FROM product_content_overrides
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (max(1, min(limit, 1000)),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_ai_fix_proposal(
    product_id: str,
    title: str,
    occupation: str,
    prior_score: int,
    payload: dict,
    status: str = "pending",
    notes: str = "",
) -> dict:
    now = utc_now_iso()
    proposal_id = str(uuid.uuid4())
    conn = db()
    # keep only one pending proposal per product
    conn.execute(
        "DELETE FROM ai_fix_proposals WHERE product_id = ? AND status = 'pending'",
        (product_id,),
    )
    conn.execute(
        """
        INSERT INTO ai_fix_proposals
        (id, product_id, title, occupation, prior_score, payload_json, status, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            proposal_id,
            product_id,
            title,
            occupation,
            int(prior_score),
            json.dumps(payload),
            status,
            notes,
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return {
        "id": proposal_id,
        "product_id": product_id,
        "title": title,
        "occupation": occupation,
        "prior_score": int(prior_score),
        "status": status,
    }


def list_ai_fix_proposals(status: str | None = "pending", limit: int = 200) -> list[dict]:
    conn = db()
    if status:
        rows = conn.execute(
            """
            SELECT id, product_id, title, occupation, prior_score, status, notes, created_at, updated_at
            FROM ai_fix_proposals
            WHERE status = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (status, max(1, min(limit, 2000))),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, product_id, title, occupation, prior_score, status, notes, created_at, updated_at
            FROM ai_fix_proposals
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 2000)),),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_ai_fix_proposal(proposal_id: str) -> dict | None:
    conn = db()
    row = conn.execute(
        """
        SELECT id, product_id, title, occupation, prior_score, payload_json, status, notes, created_at, updated_at
        FROM ai_fix_proposals
        WHERE id = ?
        """,
        (proposal_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    data = dict(row)
    try:
        data["payload"] = json.loads(data.get("payload_json") or "{}")
    except Exception:
        data["payload"] = {}
    return data


def set_ai_fix_proposal_status(proposal_id: str, status: str, notes: str = "") -> None:
    conn = db()
    conn.execute(
        """
        UPDATE ai_fix_proposals
        SET status = ?, notes = ?, updated_at = ?
        WHERE id = ?
        """,
        (status, notes, utc_now_iso(), proposal_id),
    )
    conn.commit()
    conn.close()


def apply_ai_fix_proposal(proposal_id: str) -> dict:
    proposal = get_ai_fix_proposal(proposal_id)
    if not proposal:
        raise ValueError("proposal not found")
    payload = proposal.get("payload") or {}
    title = payload.get("title") or proposal.get("title")
    if not title:
        raise ValueError("proposal missing title")
    save_content_override(
        title=title,
        category=payload.get("category") or "General",
        tagline=payload.get("tagline") or "",
        description=payload.get("description") or "",
        preview_items=payload.get("preview_items") or [],
        preview_snippet=payload.get("preview_snippet") or "",
    )
    set_ai_fix_proposal_status(proposal_id, "applied", notes="approved and applied")
    return {"ok": True, "id": proposal_id, "title": title}


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
    raw = (
        flask_request.headers.get("x-admin-token")
        or flask_request.args.get("admin_token")
        or flask_request.form.get("admin_token")
        or ""
    )
    # Be tolerant of copy/paste issues from browser or terminal.
    token = raw.strip().strip('"').strip("'").replace(" ", "+")
    return token


def admin_guard_any() -> bool:
    provided = admin_token_value()
    expected = (ADMIN_TOKEN or "").strip().replace(" ", "+")
    if provided == expected:
        return True
    # Also allow missing base64 padding in pasted tokens.
    return provided.rstrip("=") == expected.rstrip("=")


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
        "Your download links:",
    ]
    if product_id:
        product = get_product(product_id)
        if product:
            token = create_download_token({"kind": "product", "product_id": product_id, "email": email})
            experience_token = create_download_token({"kind": "experience", "product_id": product_id, "email": email})
            lines.append(
                f"- {product['title']} guided app: "
                f"{base_url}/experience/{product_id}?token={parse.quote(experience_token)}"
            )
            lines.append(
                f"- {product['title']}: {base_url}/download/product/{product_id}?token={parse.quote(token)}"
            )
    if bundle_key:
        token = create_download_token({"kind": "bundle", "bundle_key": bundle_key, "email": email})
        lines.append(
            f"- {bundle_key.replace('-', ' ').title()} bundle package: "
            f"{base_url}/download/bundle/{bundle_key}?token={parse.quote(token)}"
        )
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
    provided = (flask_request.headers.get("x-admin-token") or "").strip().replace(" ", "+")
    expected = (ADMIN_TOKEN or "").strip().replace(" ", "+")
    if provided == expected:
        return True
    return provided.rstrip("=") == expected.rstrip("=")


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

    line_y_start = 280
    line_gap = 74
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
</defs>
<rect width="1200" height="675" rx="28" fill="url(#g)"/>
<rect x="54" y="54" width="1092" height="567" rx="24" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="2"/>
<circle cx="1044" cy="114" r="78" fill="#ffffff" fill-opacity="0.10"/>
<circle cx="1005" cy="540" r="122" fill="#ffffff" fill-opacity="0.07"/>
<rect x="84" y="72" width="290" height="56" rx="28" fill="#ffffff" fill-opacity="0.18"/>
<text x="116" y="109" fill="#ffffff" font-size="34" font-family="Sora, Arial, sans-serif" font-weight="700">Northstar Studio</text>
<text x="84" y="178" fill="#ffffff" fill-opacity="0.92" font-size="38" font-family="Manrope, Arial, sans-serif">{safe_category}</text>
{line_markup}
<rect x="84" y="478" width="1032" height="2" fill="#ffffff" fill-opacity="0.28"/>
<rect x="84" y="520" width="250" height="44" rx="22" fill="#ffffff" fill-opacity="0.16" stroke="#ffffff" stroke-opacity="0.28" />
<text x="112" y="549" fill="#ffffff" font-size="24" font-family="Sora, Arial, sans-serif" font-weight="700">Instant Download</text>
<rect x="348" y="520" width="220" height="44" rx="22" fill="#ffffff" fill-opacity="0.12" stroke="#ffffff" stroke-opacity="0.20" />
<text x="375" y="549" fill="#ffffff" fill-opacity="0.95" font-size="22" font-family="Manrope, Arial, sans-serif" font-weight="700">Editable Files</text>
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


@app.get("/experience/<product_id>")
def product_experience(product_id: str):
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404
    if admin_guard_any():
        return Response(_guided_experience_html(product), mimetype="text/html")
    token = (flask_request.args.get("token") or "").strip()
    payload = parse_download_token(token)
    if not payload or payload.get("kind") != "experience" or payload.get("product_id") != product_id:
        return jsonify(
            {
                "error": "purchase required",
                "message": "Guided product experience unlocks after purchase via private buyer link.",
                "checkout_url": f"/checkout/{product_id}",
            }
        ), 403
    return Response(_guided_experience_html(product), mimetype="text/html")


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
    ai_fix_queue = list_ai_fix_proposals(status="pending", limit=100)
    overrides = list_content_overrides(limit=100)
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
        ai_fix_queue=ai_fix_queue,
        overrides=overrides,
    )


@app.get("/admin/qc")
def admin_qc_page():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    products = list_products()
    categories = sorted({p["category"] for p in products})
    selected_category = (flask_request.args.get("category") or "").strip()
    sort_key = (flask_request.args.get("sort") or "newest").strip()
    if selected_category:
        products = [p for p in products if p.get("category") == selected_category]

    if sort_key == "title_asc":
        products = sorted(products, key=lambda x: x.get("title", "").lower())
    elif sort_key == "title_desc":
        products = sorted(products, key=lambda x: x.get("title", "").lower(), reverse=True)
    elif sort_key == "price_asc":
        products = sorted(products, key=lambda x: int(x.get("price_cents", 0)))
    elif sort_key == "price_desc":
        products = sorted(products, key=lambda x: int(x.get("price_cents", 0)), reverse=True)
    else:
        sort_key = "newest"
    token = admin_token_value()
    return render_template(
        "qc.html",
        products=products,
        categories=categories,
        selected_category=selected_category,
        sort_key=sort_key,
        admin_token_hint=token,
    )


@app.get("/health")
def health():
    return jsonify({"ok": True, "time": utc_now_iso(), "active_products": count_active_products()})


@app.get("/api/products")
def api_products():
    return jsonify({"products": list_products()})


@app.get("/download/product/<product_id>")
def customer_download_product(product_id: str):
    token = (flask_request.args.get("token") or "").strip()
    payload = parse_download_token(token)
    if not payload or payload.get("kind") != "product" or payload.get("product_id") != product_id:
        return jsonify({"error": "invalid or expired download token"}), 403
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404
    package = build_customer_product_pack(product)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"{slugify(product['title'])}-northstar-studio-{stamp}.zip"
    return Response(
        package,
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/download/bundle/<bundle_key>")
def customer_download_bundle(bundle_key: str):
    token = (flask_request.args.get("token") or "").strip()
    payload = parse_download_token(token)
    if not payload or payload.get("kind") != "bundle" or payload.get("bundle_key") != bundle_key:
        return jsonify({"error": "invalid or expired download token"}), 403
    bundle = get_bundle(bundle_key)
    if not bundle:
        return jsonify({"error": "bundle not found"}), 404
    package = build_customer_bundle_pack(bundle)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"{slugify(bundle['title'])}-northstar-studio-bundle-{stamp}.zip"
    return Response(
        package,
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/admin/download/product/<product_id>.zip")
def admin_download_product_zip(product_id: str):
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    product = get_product(product_id)
    if not product:
        return jsonify({"error": "product not found"}), 404
    payload = build_customer_product_pack(product)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"{slugify(product.get('title', 'product'))}-retail-pack-{stamp}.zip"
    return Response(
        payload,
        mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/admin/download/all-qc.zip")
def admin_download_all_qc_zip():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    products = list_products()
    selected_category = (flask_request.args.get("category") or "").strip()
    if selected_category:
        products = [p for p in products if p.get("category") == selected_category]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in products:
            folder = f"{slugify(p.get('title', 'product'))}/"
            for name, content in _customer_pack_files(p):
                zf.writestr(folder + name, content)
        zf.writestr(
            "README.txt",
            f"Northstar Studio retail pack QC archive\nTotal products: {len(products)}\n"
            + (f"Category filter: {selected_category}\n" if selected_category else ""),
        )
    suffix = f"-{slugify(selected_category)}" if selected_category else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"northstar-studio-all-qc{suffix}-{stamp}.zip"
    return Response(
        buf.getvalue(),
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


@app.get("/admin/test-agent")
def admin_test_agent():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    product_id = (flask_request.args.get("product_id") or "").strip()
    title = (flask_request.args.get("title") or "").strip()
    occupation = (flask_request.args.get("occupation") or "small business owner").strip()
    product = None
    if product_id:
        product = get_product(product_id)
    elif title:
        product = get_product_by_title(title)
    if not product:
        return jsonify({"error": "product not found", "hint": "pass product_id or title"}), 404
    return jsonify(run_virtual_product_test(product, occupation))


@app.get("/admin/test-agent/all")
def admin_test_agent_all():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    occupation = (flask_request.args.get("occupation") or "small business owner").strip()
    limit = int(flask_request.args.get("limit") or "50")
    products = list_products()[: max(1, min(200, limit))]
    reports = [run_virtual_product_test(p, occupation) for p in products]
    passed = [r for r in reports if r.get("passed")]
    failed = [r for r in reports if not r.get("passed")]
    avg = int(round(sum(int(r.get("overall_score", 0)) for r in reports) / max(1, len(reports))))
    return jsonify(
        {
            "ok": True,
            "occupation": occupation,
            "count": len(reports),
            "average_score": avg,
            "passed": len(passed),
            "failed": len(failed),
            "failures": [
                {
                    "id": r["product"]["id"],
                    "title": r["product"]["title"],
                    "score": r["overall_score"],
                    "issues": r.get("issues", []),
                }
                for r in failed
            ],
            "results": reports,
        }
    )


@app.get("/admin/ai-team/report")
def admin_ai_team_report():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    occupation = (flask_request.args.get("occupation") or "small business owner").strip()
    limit = int(flask_request.args.get("limit") or "50")
    return jsonify(build_ai_team_report(occupation, limit=limit))


@app.post("/admin/ai-team/proposals/generate")
def admin_ai_team_generate_proposals():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    occupation = (
        flask_request.form.get("occupation")
        or flask_request.args.get("occupation")
        or "small business owner"
    ).strip()
    limit = int(flask_request.form.get("limit") or flask_request.args.get("limit") or "50")
    min_score = int(flask_request.form.get("min_score") or flask_request.args.get("min_score") or "80")
    max_proposals = int(flask_request.form.get("max_proposals") or flask_request.args.get("max_proposals") or "20")
    result = generate_ai_fix_proposals(occupation, limit=limit, min_score=min_score, max_proposals=max_proposals)
    if (flask_request.form.get("redirect") or "") == "1":
        token = admin_token_value()
        return redirect(
            f"/admin?admin_token={parse.quote(token)}&proposals=done&created={result.get('created_count', 0)}",
            code=302,
        )
    return jsonify(result)


@app.get("/admin/ai-team/proposals")
def admin_ai_team_list_proposals():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    status = (flask_request.args.get("status") or "pending").strip()
    if status.lower() == "all":
        status = None
    return jsonify({"ok": True, "proposals": list_ai_fix_proposals(status=status, limit=500)})


@app.post("/admin/ai-team/proposals/<proposal_id>/apply")
def admin_ai_team_apply_proposal(proposal_id: str):
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    try:
        result = apply_ai_fix_proposal(proposal_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    if (flask_request.form.get("redirect") or "") == "1":
        token = admin_token_value()
        return redirect(f"/admin?admin_token={parse.quote(token)}&applied=1", code=302)
    return jsonify(result)


@app.post("/admin/ai-team/proposals/<proposal_id>/reject")
def admin_ai_team_reject_proposal(proposal_id: str):
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    if not get_ai_fix_proposal(proposal_id):
        return jsonify({"error": "proposal not found"}), 404
    set_ai_fix_proposal_status(proposal_id, "rejected", notes="manually rejected")
    if (flask_request.form.get("redirect") or "") == "1":
        token = admin_token_value()
        return redirect(f"/admin?admin_token={parse.quote(token)}&rejected=1", code=302)
    return jsonify({"ok": True, "id": proposal_id, "status": "rejected"})


@app.post("/admin/ai-team/proposals/apply-all")
def admin_ai_team_apply_all_proposals():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    pending = list_ai_fix_proposals(status="pending", limit=500)
    applied = []
    errors = []
    for p in pending:
        try:
            applied.append(apply_ai_fix_proposal(p["id"]))
        except Exception as exc:
            errors.append({"id": p["id"], "title": p.get("title"), "error": str(exc)})
    result = {"ok": True, "applied_count": len(applied), "error_count": len(errors), "errors": errors}
    if (flask_request.form.get("redirect") or "") == "1":
        token = admin_token_value()
        return redirect(
            f"/admin?admin_token={parse.quote(token)}&applied_all={len(applied)}&apply_errors={len(errors)}",
            code=302,
        )
    return jsonify(result)


@app.get("/admin/inventory-overrides")
def admin_inventory_overrides():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"ok": True, "overrides": list_content_overrides(limit=500)})


@app.post("/admin/ai-team/autofix")
def admin_ai_team_autofix():
    if not admin_guard_any():
        return jsonify({"error": "unauthorized"}), 401
    occupation = (
        flask_request.form.get("occupation")
        or flask_request.args.get("occupation")
        or "small business owner"
    ).strip()
    limit = int(flask_request.form.get("limit") or flask_request.args.get("limit") or "50")
    min_score = int(flask_request.form.get("min_score") or flask_request.args.get("min_score") or "80")
    max_updates = int(flask_request.form.get("max_updates") or flask_request.args.get("max_updates") or "20")
    result = run_ai_team_autofix(occupation, limit=limit, min_score=min_score, max_updates=max_updates)
    if (flask_request.form.get("redirect") or "") == "1":
        token = admin_token_value()
        return redirect(
            f"/admin?admin_token={parse.quote(token)}&autofix=done&updated={result.get('updated_count', 0)}",
            code=302,
        )
    return jsonify(result)


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
