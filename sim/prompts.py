"""
Domain prompt banks + matching known-good answers.

Four domains, each a set of parameterized templates so the generated traffic is
varied (not the same string repeated). `business_policy` is deliberately tied to
the Acme Co. facts in tests/fixtures/knowledge_base.json so RAG grounding /
hallucination injection has a real context to agree with or contradict.

Each template is a dict:
    {"q": <question template>, "a": <good answer template>,
     "keys": [placeholder names], "opts": [tuples of fill values],
     "context_source": <kb source_id or None>, "expect_json": bool}

For a template with keys ["country", "capital"] and an opt ("France", "Paris"),
the fill dict is {"country": "France", "capital": "Paris"} and both `q` and `a`
are rendered with str.format(**fill). Templates with no keys/opts are fixed.
JSON answers use doubled braces ({{ }}) so str.format emits literal JSON braces.
"""

from __future__ import annotations

# ── Acme Co. ground-truth facts (mirror tests/fixtures/knowledge_base.json) ──
ACME_FACTS = {
    "policy-refunds": "Acme Co. refund policy: customers may request a full refund within 30 days of delivery, provided the item is unused and in its original packaging. Refunds are issued to the original payment method within 5 business days.",
    "policy-shipping": "Acme Co. ships domestic orders within one business day. Standard delivery arrives in three to five business days after dispatch. Expedited shipping (1-2 days) is available at checkout for an additional fee.",
    "policy-support-hours": "Acme Co. customer support operates Monday through Friday, 9 a.m. to 6 p.m. Eastern Time, excluding public holidays. Pro plan customers receive priority email support with a four-hour response target.",
    "policy-plans": "Acme Co. offers three subscription tiers: Basic, Pro, and Enterprise. The Pro plan includes priority support and advanced analytics. Annual billing receives a 20 percent discount compared to monthly billing.",
    "policy-subscription-changes": "Acme Co. subscribers can upgrade at any time with prorated charges applied immediately. Downgrades take effect at the start of the next billing cycle. Cancellations stop further charges; access continues until the end of the paid period.",
    "policy-payments": "Acme Co. accepts major credit cards, debit cards, and PayPal. Bank transfers are available only for annual enterprise contracts. A 14-day free trial of the Pro plan is available to new users with no credit card required.",
    "policy-data-retention": "Acme Co. retains account data for 30 days after account deletion to allow recovery, after which the data is permanently erased from active systems. Backups are purged within 90 days.",
    "policy-returns-process": "To return an item to Acme Co., start a return from the order page to receive a prepaid shipping label. Returns are inspected on arrival; approved refunds are processed within 5 business days of receipt.",
}


def _t(q, a, keys=None, opts=None, context_source=None, expect_json=False):
    return {
        "q": q, "a": a,
        "keys": keys or [], "opts": opts or [],
        "context_source": context_source, "expect_json": expect_json,
    }


# ── Factual QA ───────────────────────────────────────────────────────────────
FACTUAL = [
    _t("What is the capital of {country}?", "The capital of {country} is {capital}.",
       ["country", "capital"],
       [("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasília"),
        ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Norway", "Oslo"),
        ("Kenya", "Nairobi"), ("Peru", "Lima")]),
    _t("Who wrote {work}?", "{work} was written by {author}.",
       ["work", "author"],
       [("Hamlet", "William Shakespeare"), ("1984", "George Orwell"),
        ("The Odyssey", "Homer"), ("Pride and Prejudice", "Jane Austen"),
        ("War and Peace", "Leo Tolstoy")]),
    _t("How many {unit} are in a {whole}?", "There are {count} {unit} in a {whole}.",
       ["unit", "whole", "count"],
       [("days", "week", "7"), ("months", "year", "12"), ("hours", "day", "24"),
        ("sides", "hexagon", "6"), ("legs", "spider", "8")]),
    _t("What is the chemical symbol for {element}?",
       "The chemical symbol for {element} is {symbol}.",
       ["element", "symbol"],
       [("gold", "Au"), ("oxygen", "O"), ("sodium", "Na"),
        ("iron", "Fe"), ("helium", "He")]),
    _t("In what year did {event} happen?", "{event} happened in {year}.",
       ["event", "year"],
       [("the first Moon landing", "1969"), ("the fall of the Berlin Wall", "1989"),
        ("the invention of the World Wide Web", "1989")]),
    _t("What is the largest {category}?", "The largest {category} is {answer}.",
       ["category", "answer"],
       [("planet in our solar system", "Jupiter"),
        ("ocean on Earth", "the Pacific Ocean"), ("mammal", "the blue whale")]),
]

# ── Business / policy (RAG-grounded against Acme facts) ───────────────────────
BUSINESS_POLICY = [
    _t("What is Acme Co.'s refund window?",
       "Acme Co. allows a full refund within 30 days of delivery, provided the item is unused and in its original packaging.",
       context_source="policy-refunds"),
    _t("How long do refunds take to process?",
       "Approved refunds are issued to the original payment method within 5 business days.",
       context_source="policy-refunds"),
    _t("How fast is standard shipping at Acme?",
       "Standard delivery arrives three to five business days after dispatch, and orders ship within one business day.",
       context_source="policy-shipping"),
    _t("When is Acme customer support available?",
       "Support runs Monday through Friday, 9 a.m. to 6 p.m. Eastern Time, excluding public holidays.",
       context_source="policy-support-hours"),
    _t("What subscription plans does Acme offer?",
       "Acme offers three tiers: Basic, Pro, and Enterprise, with Pro adding priority support and advanced analytics.",
       context_source="policy-plans"),
    _t("What discount does annual billing get?",
       "Annual billing receives a 20 percent discount compared to monthly billing.",
       context_source="policy-plans"),
    _t("How do subscription upgrades and downgrades work at Acme?",
       "Upgrades apply immediately with prorated charges; downgrades take effect at the start of the next billing cycle.",
       context_source="policy-subscription-changes"),
    _t("What payment methods does Acme accept?",
       "Acme accepts major credit cards, debit cards, and PayPal; bank transfers are for annual enterprise contracts only.",
       context_source="policy-payments"),
    _t("Is there a free trial of the Pro plan?",
       "Yes — a 14-day free trial of the Pro plan is available to new users with no credit card required.",
       context_source="policy-payments"),
    _t("How long does Acme keep my data after I delete my account?",
       "Account data is retained for 30 days after deletion for recovery, then permanently erased; backups purge within 90 days.",
       context_source="policy-data-retention"),
    _t("How do I return an item to Acme?",
       "Start a return from the order page to get a prepaid label; approved refunds process within 5 business days of receipt.",
       context_source="policy-returns-process"),
]

# ── JSON-format tasks (structured output expected) ───────────────────────────
JSON_FORMAT = [
    _t("Return a JSON object with keys name and age for a user named {name} aged {age}.",
       '{{"name": "{name}", "age": {age}}}',
       ["name", "age"],
       [("Alice", "30"), ("Bob", "45"), ("Chen", "27"), ("Diego", "52"), ("Priya", "38")],
       expect_json=True),
    _t("Give me a JSON array of the first {n} positive integers.", "{array}",
       ["n", "array"],
       [("3", "[1, 2, 3]"), ("4", "[1, 2, 3, 4]"), ("5", "[1, 2, 3, 4, 5]")],
       expect_json=True),
    _t("Return JSON with fields product and in_stock (boolean) for {product}, which is available.",
       '{{"product": "{product}", "in_stock": true}}',
       ["product"],
       [("widget",), ("gadget",), ("sprocket",), ("gizmo",), ("bracket",)],
       expect_json=True),
    _t("Produce a JSON object mapping status to \"ok\" and code to {code}.",
       '{{"status": "ok", "code": {code}}}',
       ["code"],
       [("200",), ("201",), ("204",), ("302",)],
       expect_json=True),
]

# ── Multi-step reasoning ─────────────────────────────────────────────────────
REASONING = [
    _t("If a train travels {miles} miles in {hours} hours, what is its average speed?",
       "Average speed is {miles} divided by {hours}, which is {speed} miles per hour.",
       ["miles", "hours", "speed"],
       [("120", "2", "60"), ("150", "3", "50"), ("300", "5", "60"), ("90", "3", "30")]),
    _t("A shirt costs ${price} and is discounted {pct} percent. What is the sale price?",
       "A {pct} percent discount on ${price} gives a sale price of ${sale}.",
       ["price", "pct", "sale"],
       [("40", "25", "30"), ("100", "10", "90"), ("60", "50", "30"), ("80", "75", "20")]),
    _t("If {a} workers finish a job in {days} days, how long for twice as many workers?",
       "Doubling the workers halves the time, so it takes {half} days.",
       ["a", "days", "half"],
       [("4", "8", "4"), ("6", "12", "6"), ("10", "20", "10")]),
    _t("What comes next in the sequence {seq}?",
       "The pattern increases by a fixed step, so the next number is {next}.",
       ["seq", "next"],
       [("2, 4, 6, 8", "10"), ("5, 10, 15, 20", "25"),
        ("3, 6, 9, 12", "15"), ("1, 4, 7, 10", "13")]),
    _t("If today is {day}, what day is it in two days?", "Two days after {day} is {answer}.",
       ["day", "answer"],
       [("Monday", "Wednesday"), ("Friday", "Sunday"),
        ("Saturday", "Monday"), ("Tuesday", "Thursday")]),
]

BANKS = {
    "factual": FACTUAL,
    "business_policy": BUSINESS_POLICY,
    "json_format": JSON_FORMAT,
    "reasoning": REASONING,
}
