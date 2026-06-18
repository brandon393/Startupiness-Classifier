"""
Stage 3: Full LLM Classification Pipeline
==========================================
- Fetches every company from Searchlight via MCP
- Sends ALL companies to OpenAI with the full Startupiness rules
- Rules engine used only as fallback if OpenAI call fails

Usage:
    python Startupiness_LLM_Classifier.py --stream 78A110D63ED9EAB
    python Startupiness_LLM_Classifier.py --list 966DB77ED88BB50
    python Startupiness_LLM_Classifier.py --domains stripe.com notion.so linear.app
    python Startupiness_LLM_Classifier.py --summary

Requirements:
    pip install requests python-dotenv openai

.env file:
    SEARCHLIGHT_TOKEN=mcp_...
    OPENAI_API_KEY=sk-...
"""

import os, json, time, argparse, requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

SEARCHLIGHT_TOKEN = os.getenv("SEARCHLIGHT_TOKEN")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
MCP_URL           = "https://searchlight.headline.com/api/mcp"
OPENAI_MODEL      = "gpt-5.4-mini"
RATE_LIMIT_DELAY  = 0.2   # seconds between OpenAI calls

if not SEARCHLIGHT_TOKEN:
    print("❌  SEARCHLIGHT_TOKEN not set in .env"); import sys; sys.exit(1)

if not OPENAI_API_KEY:
    print("❌  OPENAI_API_KEY not set in .env"); import sys; sys.exit(1)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── System Prompt (full Notion doc rules) ─────────────────────────────────────

SYSTEM_PROMPT = """You are a startup classification engine for Headline, a venture capital firm.

Given structured data about a company from Searchlight, classify it into exactly ONE category.

═══════════════════════════════════════
CLASSIFICATION TIERS — pick exactly ONE
═══════════════════════════════════════

Pick exactly ONE output tag:

  early        — Investable startup, pre-Series A, under $30M total funding.
                 Builds scalable tech product/platform. Could be venture-backable.
                 Includes seed, pre-seed, and unfunded companies with strong product signals.

  growth       — Tech company at growth stage. Requires Series A or later round
                 OR total funding >= $30M. Must show clear product/tech focus and
                 be actively operating (not stale or acquired).

  established  — Large established company, public or effectively institutional.
                 Not investable at our stage. Includes subsidiaries of large corporations,
                 acquired companies, IPO'd companies.

  non-startup  — Not investable. Includes:
                 • SMALL_BUSINESS (local/niche, no scalability, traditional model)
                 • AGENCY (sells services, no proprietary product)
                 • PERSONAL_SITE (portfolio, blog, personal brand)
                 • OTHER (media outlet, nonprofit, open-source without commercial entity)

═══════════════════════════════
LLM RULES (apply in this order)
═══════════════════════════════

Parent company / subsidiary rule (check FIRST before anything else):
- If the company is a SUBSIDIARY of a large established corporation → established
- If the company is a DIVISION or BRAND of a public company → established
- Keywords: "part of [group]", "subsidiary of", "owned by", "a [Corp] company" → established
- If the parent is publicly traded, a bank, telco, major conglomerate, or government → established
- Exception: if the subsidiary has its own VC funding and startup characteristics, classify on its own merits

Corporate open source rule (check SECOND):
- If the product is an open source project CREATED BY or MAINTAINED BY a large corporation
  (Google, Microsoft, Meta, Apple, Amazon, etc.) → established
- Even if the project has its own domain, small listed team, or strong GitHub signals,
  it is NOT an investable startup if it is backed by a large corporation
- Examples: go.dev (Google), playwright.dev (Microsoft), react.dev (Meta) → all established
- Signal: web search or Attio comments mention a big tech company as creator/maintainer

Tie-breaking:
- When uncertain between early/growth and non-startup → choose early or growth
- When uncertain between early and AGENCY              → choose early
- When uncertain between early and SMALL_BUSINESS      → choose early
- Non-traditional industry + tech differentiator    = early (e.g. Kalshi, Character.ai)
- Open-source project with commercial entity        = early
- Has /pricing, /docs, /api, /changelog             = strong product signal → early or growth
- "We build for clients" + no self-serve product    = non-startup
- No visible revenue but clear tech development     = early
- Hybrid services + product → early/growth if product is primary focus

Industry soft excludes (override if product signals are present):
- Staffing / recruitment agency (non-tech)
- Marketing / design agency
- IT consulting / dev shop
- News / media outlet

Team / Founder positive signals (weight toward early/growth):
- Technical founders (engineering, CS background)
- Small team with senior titles (CTO, VP Eng on a 5-person team)
- Hiring engineering-heavy (>50% of open roles are technical)
- Team grew recently (QoQ headcount increase)

═══════════════════════════════
STAGE GUIDANCE
═══════════════════════════════

Use these signals to decide between early and growth:
  early  — pre-Series A AND total funding under $30M. Includes seed, pre-seed,
            or unfunded startups with strong product signals.
            Headcount alone does NOT trigger growth.

  growth — Series A or later round
            OR total funding >= $30M (regardless of round type)
            OR clear evidence of proven revenue at scale + still actively growing

CRITICAL: ANY ONE of these alone triggers growth:
- Total funding >= $30M → growth
- Series A or later round → growth

IMPORTANT: Seed and pre-seed under $30M = early.
Seed or pre-seed with $30M+ total raised = growth.
Series A regardless of amount = growth.
Headcount >= 50 is a signal, not a trigger. Weigh it alongside funding stage and revenue evidence.
A 15-year-old services firm with 60 employees is non-startup, not growth.
A 3-year-old Series A company with $10M raised is growth, not early.

═══════════════════════════════
OUTPUT FORMAT
═══════════════════════════════

Respond ONLY with valid JSON. No markdown, no explanation, nothing else:
{
  "classification": "early",
  "confidence": 0.85,
  "rationale": "Has self-serve pricing, API docs, technical founder from Google, 8-person team. No funding data but strong product signals.",
  "signals": ["pricing_page", "api_docs", "technical_founder", "small_team", "growing_headcount"]
}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_funding(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip().upper().replace(",", "")
    for suffix, mult in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if value.endswith(suffix):
            try:
                return float(value[:-1]) * mult
            except ValueError:
                return None
    try:
        return float(value)
    except ValueError:
        return None

def parse_year(value):
    if value is None:
        return None
    try:
        return int(str(value)[:4])
    except (ValueError, TypeError):
        return None

def format_funding(value):
    f = parse_funding(value)
    if f is None:
        return "unknown"
    if f >= 1e9:
        return f"${f/1e9:.1f}B"
    if f >= 1e6:
        return f"${f/1e6:.1f}M"
    return f"${f:,.0f}"

# ── Searchlight MCP Client ────────────────────────────────────────────────────

class SearchlightMCP:
    def __init__(self, token):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        })

    def _call(self, tool, arguments):
        payload = {
            "jsonrpc": "2.0",
            "method":  "tools/call",
            "id":      1,
            "params":  {"name": tool, "arguments": arguments},
        }
        try:
            r = self.session.post(MCP_URL, json=payload, timeout=20)
            if r.status_code == 401:
                print("\n❌  Token expired. Get a new one from:")
                print("    https://searchlight.headline.com/settings → MCP Server → API Tokens")
                import sys; sys.exit(1)
            if r.status_code != 200:
                return None
            data    = r.json()
            content = data.get("result", {}).get("content", [])
            for block in content:
                if block.get("type") == "text":
                    return json.loads(block["text"])
            return None
        except Exception as e:
            print(f"    MCP error: {e}")
            return None

    def iter_list(self, access_token):
        """Yield every company in a Searchlight list, paginating automatically."""
        page        = 0
        total_shown = False
        per_page    = 50

        while True:
            data = self._call("get_list_companies", {
                "access_token":   access_token,
                "page":           page,
                "per_page":       per_page,
                "include_details": True,
            })
            if data is None:
                print(f"  ✗ Failed to fetch page {page}")
                break

            # List API returns an array directly, not a dict with data/results
            results = data if isinstance(data, list) else (data.get("data") or data.get("results") or [])

            if not total_shown:
                total = len(results) if isinstance(data, list) else data.get("total", "?")
                print(f"List {access_token}: fetching companies...")
                total_shown = True

            if not results:
                break

            yield from results

            # Lists paginate differently — stop if we got fewer than per_page
            if len(results) < per_page:
                break

            page += 1

    def get_company(self, domain):
        result = self._call("lookup_company", {"domain": domain})
        if result is None:
            return None
        return result[0] if isinstance(result, list) else result

    def iter_stream(self, access_token):
        page        = 0
        total_shown = False
        per_page    = 10

        while True:
            data = self._call("get_stream_companies", {
                "access_token":    access_token,
                "page":            page,
                "per_page":        per_page,
                "include_details": True,
            })
            if data is None:
                print(f"  ✗ Failed to fetch page {page}")
                break

            results = data.get("data") or data.get("results") or []

            if not total_shown:
                print(f"Stream {access_token}: {data.get('total', '?')} companies.")
                total_shown = True

            if not results:
                break

            yield from results

            if not data.get("has_more", False):
                break

            page += 1
            time.sleep(0.1)

# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(company):
    """Convert Searchlight company data into a structured prompt for the LLM."""
    domain   = company.get("domain", "unknown")
    name     = company.get("name", domain)
    desc     = company.get("description") or "No description available."
    hc       = company.get("employee_count")
    hc_6m    = company.get("employee_percent_growth_last_6mo")
    hc_3m    = company.get("employee_percent_growth_last_3mo")
    funding  = company.get("funding_total")
    founded  = company.get("founding_date")
    country  = company.get("country")
    cats     = ", ".join(company.get("categories") or []) or "none"
    sl_score = company.get("searchlight_score")
    pedigree = company.get("seed_investor_pedigree")

    # Format funding rounds
    rounds = company.get("funding_rounds") or []
    if rounds:
        lines = []
        for r in rounds[-5:]:
            investors = ", ".join(r.get("lead_investors") or []) or "unknown"
            lines.append(
                f"  • {r.get('type','?')}  |  {r.get('date','?')}  |  investors: {investors}"
            )
        rounds_str = "\n".join(lines)
    else:
        rounds_str = "  None on record"

    # Extract Attio revenue signals from CRM comments
    attio       = company.get("attio") or {}
    attio_lines = []
    for comment in (attio.get("recent_comments") or []):
        text = (comment.get("content") or "").strip()
        if text:
            attio_lines.append(f"  • {text}")
    attio_str = "\n".join(attio_lines) if attio_lines else "  None on record"

    return f"""Classify this company:

DOMAIN:       {domain}
NAME:         {name}
DESCRIPTION:  {desc}

TEAM:
  Headcount:          {hc if hc is not None else 'unknown'}
  Growth (6 months):  {f'{hc_6m:.1%}' if hc_6m is not None else 'unknown'}
  Growth (3 months):  {f'{hc_3m:.1%}' if hc_3m is not None else 'unknown'}

COMPANY INFO:
  Founded:            {founded or 'unknown'}
  Country:            {country or 'unknown'}
  Categories:         {cats}

FUNDING:
  Total raised:       {format_funding(funding)}
  Round history:
{rounds_str}

SIGNALS:
  Investor pedigree:  {f'{pedigree:.2f}' if pedigree is not None else 'unknown'}  (0=none, 1=top-tier VC)
  Searchlight score:  {sl_score or 'unknown'}  (0-100, higher = healthier growth)

REVENUE / CRM NOTES (internal deal notes — treat as high-confidence signals):
{attio_str}

IMPORTANT: Before classifying, check if this company could be a subsidiary \
or division of a larger established corporation. Look for hints in the name \
and description such as 'part of X group', 'a X company', 'subsidiary of X', \
'owned by X', 'member of X group'. If the parent is a large corporation, bank, \
telco, or public company, classify as PUBLIC_OR_ESTABLISHED.

Using the classification rules, classify this company now."""

# ── OpenAI Classifier ─────────────────────────────────────────────────────────

class OpenAIClassifier:
    def __init__(self, client):
        self.client = client

    # Patterns that suggest a company might be a subsidiary or acquired
    OWNERSHIP_PHRASES = [
        "subsidiary of", "division of", "part of", "acquired by",
        "owned by", "a member of", "wholly owned", "brand of",
        "joint venture with", "member of the",
    ]

    FINANCIAL_DOMAIN_PATTERNS = [
        "pay", "bank", "financial", "capital", "wealth", "invest",
        "insurance", "assurance", "lending", "credit", "fund",
        "money", "cash", "payment", "fintech", "trading",
    ]

    def _needs_subsidiary_check(self, company, first_result):
        """
        Returns True only when there is a strong signal this company
        might be a subsidiary or acquired — triggering a web search pass.

        Five targeted triggers:
        1. LLM said early/growth BUT confidence < 80% AND domain has financial patterns
        2. Description OR Attio comments contain ownership language or big tech company names
        3. Searchlight acquired flag is set
        4. Tiny + old + no funding — likely dead or acquired (e.g. cradleaccounting.com)
        5. Ambiguous consulting/services description — verify actual product with web search
        """
        # Trigger 1: Low confidence early/growth with financial domain pattern
        cls  = (first_result.get("classification") or "").lower()
        conf = first_result.get("confidence", 1.0)
        if cls in ("early", "growth") and conf < 0.80:
            domain_base = (company.get("domain") or "").lower()
            domain_base = domain_base.replace(".com","").replace(".io","").replace(".ai","")\
                                     .replace(".co","").replace(".net","").replace(".org","")\
                                     .replace(".ca","").replace(".app","").replace(".xyz","")
            if any(p in domain_base for p in self.FINANCIAL_DOMAIN_PATTERNS):
                return True

        # Trigger 2: Explicit ownership language in description OR Attio comments
        desc = (company.get("description") or "").lower()
        if any(phrase in desc for phrase in self.OWNERSHIP_PHRASES):
            return True

        # Also check Attio comments for ownership language or big tech company names
        BIG_TECH = [
            "google", "microsoft", "meta", "apple", "amazon", "salesforce",
            "oracle", "sap", "ibm", "adobe", "intuit", "workday", "servicenow",
            "wells fargo", "jpmorgan", "goldman sachs", "morgan stanley",
            "blackrock", "visa", "mastercard", "paypal", "stripe",
        ]
        attio_comments = company.get("attio", {}).get("recent_comments", [])
        for comment in attio_comments:
            content = (comment.get("content") or "").lower()
            if any(phrase in content for phrase in self.OWNERSHIP_PHRASES):
                return True
            if any(tech in content for tech in BIG_TECH):
                return True

        # Trigger 3: Searchlight acquired flag
        if company.get("acquired"):
            return True

        # Trigger 4: Tiny + old + no funding → likely acquired or dead
        hc      = company.get("employee_count")
        funding = parse_funding(company.get("funding_total"))
        founded = parse_year(company.get("founding_date"))
        if (hc is not None and hc <= 5
                and (funding is None or funding == 0)
                and founded is not None and (2026 - founded) >= 4):
            return True

        # Trigger 5: Ambiguous consulting/services description → verify with web search
        # Skip if it looks like a personal site (personal TLD patterns or personal name domains)
        domain = (company.get("domain") or "").lower()
        PERSONAL_SITE_SIGNALS = [
            "zx2c4", "github.io", "gitlab.io", "netlify.app", "vercel.app",
        ]
        is_personal = (
            any(s in domain for s in PERSONAL_SITE_SIGNALS)
            or (company.get("employee_count") or 0) <= 1
        )
        AMBIGUOUS_DESC_PATTERNS = [
            "it company", "it consulting", "software development", "consulting",
            "solutions provider", "it services", "technology solutions",
            "software solutions", "digital solutions", "tech solutions",
            "provides software", "provides services", "technology services",
            "managed services", "service provider", "custom software",
        ]
        if not is_personal and any(phrase in desc for phrase in AMBIGUOUS_DESC_PATTERNS):
            return True

        return False

    def _classify_standard(self, company):
        """Standard classification using Chat Completions."""
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            max_completion_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(company)},
            ],
        )
        raw = response.choices[0].message.content.strip()
        return raw, "openai"

    def _classify_with_websearch(self, company):
        """
        Re-classify using Responses API with web search.
        Only triggered for suspected subsidiaries/acquired companies.
        """
        domain = company.get("domain", "")
        try:
            response = self.client.responses.create(
                model=OPENAI_MODEL,
                tools=[{"type": "web_search"}],
                instructions=SYSTEM_PROMPT,
                input=(
                    f"Search the web to check whether '{domain}' is a subsidiary "
                    f"or was acquired by a larger corporation. "
                    f"If it is a subsidiary or acquired company, classify as established. "
                    f"Otherwise classify normally using the rules.\n\n"
                    + build_prompt(company)
                ),
            )
            raw = ""
            for item in response.output:
                if hasattr(item, "content"):
                    for block in item.content:
                        if hasattr(block, "text"):
                            raw = block.text
                            break
                if raw:
                    break
            if not raw and hasattr(response, "output_text"):
                raw = response.output_text
            return raw, "openai+websearch"
        except Exception as e:
            print(f"\n    ⚠️  Web search failed for {domain}, using standard result: {e}")
            return None, None

    def classify(self, company):
        try:
            # Step 1: Standard classification
            raw, classified_by = self._classify_standard(company)
            raw    = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            result["classified_by"] = classified_by

            # Step 2: Check if we should re-classify with web search
            if self._needs_subsidiary_check(company, result):
                print(f"    🔍 (subsidiary check)", end=" ", flush=True)
                ws_raw, ws_by = self._classify_with_websearch(company)
                if ws_raw:
                    ws_raw   = ws_raw.replace("```json", "").replace("```", "").strip()
                    ws_result = json.loads(ws_raw)
                    ws_result["classified_by"] = ws_by
                    return ws_result

            return result
        except Exception as e:
            print(f"\n    ⚠️  OpenAI error for {company.get('domain')}: {e}")
            return None
# ── Fallback Rules Classifier ─────────────────────────────────────────────────

TECH_KEYWORDS = {
    "saas", "software", "artificial intelligence", "machine learning",
    "developer tools", "api", "cloud", "cybersecurity", "fintech",
    "healthtech", "biotech", "blockchain", "marketplace", "analytics",
    "data", "infrastructure", "devops", "platform", "automation",
    "robotics", "climate tech", "hardware", "iot", "productivity",
    "payments", "security", "database", "crypto", "edtech",
}

AGENCY_KEYWORDS = [
    "consulting", "we build for", "build for clients", "custom development",
    "outsourcing", "digital marketing", "creative agency", "design agency",
    "marketing agency", "it services", "managed services", "staffing",
    "recruitment agency", "dev shop", "service provider",
]

GROWTH_ROUND_TYPES = {"series a", "series b", "series c", "series d", "series e"}
EARLY_ROUND_TYPES  = {"seed", "pre-seed", "pre seed", "angel", "convertible note"}

def fallback_rules(company):
    """Simple fallback if OpenAI fails."""
    desc     = (company.get("description") or "").lower()
    hc       = company.get("employee_count")
    funding  = parse_funding(company.get("funding_total"))
    rounds   = company.get("funding_rounds") or []
    round_types = {(r.get("type") or "").lower() for r in rounds}

    has_growth = any(any(g in rt for g in GROWTH_ROUND_TYPES) for rt in round_types)
    has_early  = any(any(e in rt for e in EARLY_ROUND_TYPES)  for rt in round_types)
    matched_tech = {t for t in TECH_KEYWORDS if t in desc}
    agency_hits  = [kw for kw in AGENCY_KEYWORDS if kw in desc]

    if len(agency_hits) >= 2 and not matched_tech:
        return {"classification": "AGENCY", "stage": None, "confidence": 0.70,
                "rationale": "Fallback: agency signals in description.", "signals": agency_hits[:2],
                "classified_by": "fallback"}

    stage = "GROWTH" if (has_growth or (funding and funding >= 30e6)) else "EARLY_STAGE"
    return {"classification": "early" if stage == "EARLY_STAGE" else "growth", "confidence": 0.60,
            "rationale": "Fallback: OpenAI unavailable, defaulting to early/growth.",
            "signals": [], "classified_by": "fallback"}

# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(stream_token=None, list_token=None, domains=None, output_path="results.jsonl", resume=True):
    sl  = SearchlightMCP(SEARCHLIGHT_TOKEN)
    llm = OpenAIClassifier(openai_client)

    # Resume: skip already-classified domains
    processed = set()
    if resume and os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["domain"])
                except Exception:
                    pass
        if processed:
            print(f"Resuming — skipping {len(processed)} already classified.")

    # Fetch companies
    if stream_token:
        companies = list(sl.iter_stream(stream_token))
    elif list_token:
        companies = list(sl.iter_list(list_token))
    elif domains:
        companies = []
        for d in domains:
            print(f"  Fetching {d}...", end=" ", flush=True)
            c = sl.get_company(d)
            if c:
                companies.append(c); print("✓")
            else:
                print("✗ not found")
    else:
        raise ValueError("Provide --stream, --list, or --domains")

    stats = {"total": 0, "skipped": 0, "openai": 0, "fallback": 0}

    with open(output_path, "a") as out:
        for company in companies:
            domain = company.get("domain", "unknown")
            stats["total"] += 1

            if domain in processed:
                stats["skipped"] += 1
                continue

            # Try OpenAI first
            result = llm.classify(company)

            # Fallback to rules if OpenAI fails
            if result is None:
                result = fallback_rules(company)
                stats["fallback"] += 1
            else:
                stats["openai"] += 1

            result["domain"]        = domain
            result["name"]          = company.get("name", domain)
            result["classified_at"] = datetime.now(timezone.utc).isoformat()

            cls   = result["classification"]
            conf  = result.get("confidence", 0)
            print(f"  {domain:<40}  {cls:<15}  [{conf:.0%}]")

            out.write(json.dumps(result) + "\n")
            out.flush()
            time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone — total: {stats['total']} | openai: {stats['openai']} | fallback: {stats['fallback']} | skipped: {stats['skipped']}")
    print(f"Output: {output_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(output_path):
    from collections import Counter
    counts = Counter()
    by_engine = Counter()
    total  = 0
    with open(output_path) as f:
        for line in f:
            try:
                row   = json.loads(line)
                cls   = row.get("classification", "UNKNOWN")
                counts[cls] += 1
                by_engine[row.get("classified_by", "?")] += 1
                total += 1
            except Exception:
                pass
    if not total:
        print("No results yet."); return
    print(f"\n{'─'*52}\nRESULTS ({total} companies)\n{'─'*52}")
    for label, n in counts.most_common():
        bar = "█" * int(n / total * 30)
        print(f"  {label:<40}  {n:>4}  {bar}")
    print(f"{'─'*52}")
    for engine, n in by_engine.most_common():
        print(f"  Classified by {engine:<10} {n:>4}")
    print(f"{'─'*52}\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 3 Full LLM Classifier")
    p.add_argument("--stream",    help="Any Searchlight stream access token")
    p.add_argument("--list",      help="Any Searchlight list access token")
    p.add_argument("--domains",   nargs="+", help="Domains to classify directly")
    p.add_argument("--output",    default="results.jsonl")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--clear",     action="store_true", help="Delete previous results before running")
    p.add_argument("--summary",   action="store_true")
    args = p.parse_args()

    if args.clear and os.path.exists(args.output):
        os.remove(args.output)
        print(f"Cleared previous results: {args.output}")

    if args.summary:
        print_summary(args.output)
    else:
        run_pipeline(
            stream_token=args.stream,
            list_token=getattr(args, 'list', None),
            domains=args.domains,
            output_path=args.output,
            resume=not args.no_resume and not args.clear,
        )
        print_summary(args.output)
