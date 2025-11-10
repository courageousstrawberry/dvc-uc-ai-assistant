# ai_agent.py — DVC → UC transfer assistant (LLM-centric parsing + multi-turn)
# Scope: Transfer-only; Campuses: UCB / UCD / UCSD
# Adds: Multi-campus selection + Category filtering (to merge Dani's + Eleni's approaches)

import os, json, glob, re, csv, argparse, uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dotenv import load_dotenv
from openai import OpenAI

#campus config (3 only)
CAMPUS_ALIASES = {
    "UCB": ["uc berkeley", "berkeley", "ucb", "cal"],
    "UCD": ["uc davis", "davis", "ucd"],
    "UCSD": ["uc san diego", "san diego", "ucsd"],
}
PRETTY_CAMPUS = {
    "UCB": "UC Berkeley",
    "UCD": "UC Davis",
    "UCSD": "UC San Diego",
}

#common typo fixes
TYPO_FIXES = {
    r"\busb\b": "uc berkeley",
    r"\bucb\b": "uc berkeley",
    r"\bberkley\b": "berkeley",
    r"\bucsd\b": "uc san diego",
    r"\buc sd\b": "uc san diego",
}

def normalize_typos(q: str) -> str:
    t = q.lower()
    for pat, repl in TYPO_FIXES.items():
        t = re.sub(pat, repl, t)
    return t

def detect_campus_from_query(q: str) -> Optional[str]:
    t = normalize_typos(q)
    for key, aliases in CAMPUS_ALIASES.items():
        if any(a in t for a in aliases):
            return key
    return None

#new: detect multiple campuses
def detect_campuses_from_query(q: str) -> List[str]:
    t = normalize_typos(q)
    found: List[str] = []
    for key, aliases in CAMPUS_ALIASES.items():
        if any(a in t for a in aliases):
            found.append(key)
    return sorted(set(found))

#category detection (Eleni's pathway):
CATEGORY_ALIASES = {
    #high-level groupings often seen in assist/grids
    "major preparation": ["major preparation", "lower division major", "ld major"],
    "lower division major": ["lower division major", "ld major"],
    "general education": ["general education", "ge", "breadth"],
    "breadth": ["breadth", "ge area", "area"],
    "math": ["math", "mathematics"],
    "science": ["science", "natural science", "biology", "chemistry", "physics"],
    "computer science": ["computer science", "cs", "programming", "software"],
    #add more if your JSON uses other labels (e.g., "Engineering Fundamentals", "Major Requirements")
}

def _canon_category_tokens() -> List[str]:
    #return unique canonical keys 
    return list(CATEGORY_ALIASES.keys())

def normalize_categories_freeform(text: str) -> List[str]:
    """
    Pull category intents from free text.
    Strategy:
      1) If user writes: category: "<something>", capture inside quotes.
      2) If user says "only <phrase>" or "show <phrase> only", capture phrase.
      3) Fuzzy match known aliases and canonical keys.
    """
    low = normalize_typos(text)
    picked: Set[str] = set()

    # 1)category: "..."
    for m in re.finditer(r'category\s*:\s*["“](.+?)["”]', low):
        phrase = m.group(1).strip()
        if phrase:
            picked.add(phrase)

    # 2)"only <phrase>" / "show <phrase> only"
    for m in re.finditer(r'\bonly\s+([a-z0-9 \-/&]+)', low):
        phrase = m.group(1).strip()
        
        phrase = re.split(r"[.,;:!?()\[\]{}]", phrase)[0].strip()
        if phrase:
            picked.add(phrase)

    for m in re.finditer(r'\bshow\s+([a-z0-9 \-/&]+?)\s+only\b', low):
        phrase = m.group(1).strip()
        if phrase:
            picked.add(phrase)

    # 3)fuzzy match against known aliases/keys
    for canon, variants in CATEGORY_ALIASES.items():
        if canon in low:
            picked.add(canon)
            continue
        for v in variants:
            if v in low:
                picked.add(canon)
                break

    #normalize: keep canonical keys if they match, otherwise keep raw phrase 
    out: Set[str] = set()
    canon_keys = set(_canon_category_tokens())
    for p in picked:
        
        matched_key = None
        for ck in canon_keys:
            if ck in p:
                matched_key = ck
                break
        out.add(matched_key or p)
    return sorted(out)

#local helpers 
def _normalize_single_code(raw: str) -> str:
    s = raw.upper().strip().replace(" ", "-")
    if s.startswith(("CS-", "COMPSCI-", "COMSCI-", "COMPSC-")):
        s = "COMSC-" + s.split("-", 1)[1]
    m = re.match(r"^([A-Z&]+)[- ]?(\d+[A-Z]?)$", s)
    if m:
        s = f"{m.group(1)}-{m.group(2)}"
    return s

def parse_completed_freeform(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for code in re.findall(r"\b([A-Za-z]{2,}[- ]?\d+[A-Za-z]?)\b", text, flags=re.IGNORECASE):
        tokens.add(_normalize_single_code(code))
    return tokens

def parse_preferences_seed(q: str) -> dict:
    t = " " + q.lower().strip() + " "
    want_cs = any(x in t for x in [" cs ", "comsc", "computer science", "programming", "data structures"])
    want_math = any(x in t for x in [" math ", "calculus", "linear algebra", "differential equations"])
    want_science = any(x in t for x in [" science ", "physics", "chemistry", "biology", " bio ", " chem ", " phys "])
    exclusive_domain = None
    if want_cs and not (want_math or want_science):
        exclusive_domain = "cs"
    elif want_math and not (want_cs or want_science):
        exclusive_domain = "math"
    elif want_science and not (want_cs or want_math):
        exclusive_domain = "science"

    #new: seed categories 
    seed_categories = normalize_categories_freeform(q)

    return {
        "required_only": any(x in t for x in ["required only", "only required", "must have", "need all"]),
        "exclusive_domain": exclusive_domain,
        "want_cs": want_cs,
        "want_math": want_math,
        "want_science": want_science,
        "seed_categories": seed_categories,  
    }

#load/collect/filter data
def load_all_data(paths: List[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for pattern in paths:
        for path in glob.glob(pattern):
            base = os.path.basename(path)
            campus_key = base.split("_")[0].upper()
            if campus_key not in PRETTY_CAMPUS:
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data[campus_key] = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return data

def collect_course_rows(campus_json: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def _rec(o: Any):
        if isinstance(o, dict):
            if "Category" in o and "Courses" in o:
                cat = o.get("Category", "")
                mr = o.get("Minimum_Required", "")
                courses = o.get("Courses", [])
                if isinstance(courses, list):
                    for pair in courses:
                        dvc_block = pair.get("DVC")
                        items = dvc_block if isinstance(dvc_block, list) else [dvc_block]
                        for d in items:
                            if isinstance(d, dict):
                                out.append({
                                    "category": cat,
                                    "minimum_required": mr,
                                    "dvc_code": d.get("Course_Code", "") or d.get("Code", ""),
                                    "dvc_title": d.get("Title", ""),
                                    "dvc_units": d.get("Units", "") or d.get("units", ""),
                                })
            for v in o.values():
                _rec(v)
        elif isinstance(o, list):
            for i in o:
                _rec(i)

    _rec(campus_json)

    #dedupe by code
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for r in out:
        code = (r.get("dvc_code") or "").strip()
        if code and code not in seen:
            dedup.append(r)
            seen.add(code)
    return dedup

def is_cs_row(row: dict) -> bool:
    code = (row.get("dvc_code") or "").upper()
    title = (row.get("dvc_title") or "").lower()
    cat = (row.get("category") or "").lower()
    return (
        code.startswith(("COMSC-", "COMSCI-", "COMPSC-", "CS-"))
        or "programming" in title
        or "data structures" in title
        or "software" in title
        or "major preparation" in cat
        or "lower division major" in cat
        or "computer science" in cat
    )

def is_math_row(row: dict) -> bool:
    code = (row.get("dvc_code") or "").upper()
    cat = (row.get("category") or "").lower()
    title = (row.get("dvc_title") or "").lower()
    return (
        code.startswith(("MATH-", "STAT-"))
        or "mathematics" in cat
        or "math" in cat
        or "calculus" in title
        or "linear algebra" in title
        or "differential equations" in title
    )

def is_science_row(row: dict) -> bool:
    code = (row.get("dvc_code") or "").upper()
    cat = (row.get("category") or "").lower()
    return (
        code.startswith(("PHYS-", "CHEM-", "BIOSC-", "BIOL-"))
        or "physics" in cat
        or "chemistry" in cat
        or "biology" in cat
        or "science" in cat
    )

def _row_matches_any_category(row: dict, categories: List[str]) -> bool:
    if not categories:
        return True
    cat_text = (row.get("category") or "").lower()
    if not cat_text:
        return False
    for requested in categories:
        rlow = requested.lower()
        #if the request is a canonical key, check aliases aswell
        if rlow in CATEGORY_ALIASES:
            if any(alias in cat_text for alias in CATEGORY_ALIASES[rlow] + [rlow]):
                return True
        #otherwise do substring match on provided phrase
        if rlow in cat_text:
            return True
    return False

def filter_rows(rows: List[Dict[str, Any]],
                seed_prefs: dict,
                completed_courses: Set[str],
                completed_domains: Set[str],
                focus_only: Optional[str],
                required_only: bool,
                categories_only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Apply filters in priority:
    1) Remove completed courses and completed domains
    2) Apply focus_only (cs|math|science)
    3) Apply required_only (Minimum_Required is 'all' or positive int)
    4) Apply categories_only (string-contains on row['category'])
    5) Otherwise allow mixed
    """
    filtered: List[Dict[str, Any]] = []
    completed_upper = {c.upper() for c in completed_courses}

    #seed categories act as hints if categories_only is None
    if categories_only is None:
        categories_only = seed_prefs.get("seed_categories") or []

    for r in rows:
        code = (r.get("dvc_code") or "").upper()

        #filter out completed specific courses
        if code in completed_upper:
            continue

        #filter out completed domains
        if "science" in completed_domains and is_science_row(r):
            continue
        if "math" in completed_domains and is_math_row(r):
            continue
        if "cs" in completed_domains and is_cs_row(r):
            continue

        #focus filter (LLM-driven)
        if focus_only == "cs" and not is_cs_row(r):
            continue
        if focus_only == "math" and not is_math_row(r):
            continue
        if focus_only == "science" and not is_science_row(r):
            continue

        #required-only filter
        if required_only:
            mr = str(r.get("minimum_required", "")).lower()
            if not (mr == "all" or (mr.isdigit() and int(mr) > 0)):
                continue

        #seed prefs domain-exclusive
        exclusive = seed_prefs.get("exclusive_domain")
        if exclusive == "cs" and not is_cs_row(r):
            continue
        if exclusive == "math" and not is_math_row(r):
            continue
        if exclusive == "science" and not is_science_row(r):
            continue

        #new: categories filter (Leni merge)
        if not _row_matches_any_category(r, categories_only):
            continue

        filtered.append(r)
    return filtered

#LLM: single-turn structured parser
def llm_parse_user_message(client: OpenAI, user_message: str) -> Dict[str, Any]:
    """
    Return JSON like:
    {
      "intent": "find_requirements" | "find_equivalent_course",
      "parameters": {
        "campus": "UCB|UCD|UCSD" | null,
        "campuses": ["UCB","UCD"] | [],
        "target_course_code": "CS 61A" | null,
        "target_institution": "UC Berkeley" | null
      },
      "filters": {
        "focus_only": "cs|math|science|all|null",
        "required_only": true|false,
        "domains_completed": ["science","math","cs"],
        "completed_courses": ["COMSC-110", "MATH-192"],
        "categories": ["major preparation","breadth"]   # NEW
      }
    }
    """
    system = (
        "You are an assistant that parses TRANSFER-ONLY student questions for UC transfer planning from Diablo Valley College (DVC). "
        "Output STRICT JSON (no markdown, no commentary). Keys: intent, parameters, filters.\n"
        "Allowed intents: find_requirements, find_equivalent_course.\n"
        "parameters.campus: normalize to UCB, UCD, or UCSD when possible (else null) for backward compatibility.\n"
        "parameters.campuses: ARRAY of campuses (UCB, UCD, UCSD) if multiple are requested (else empty).\n"
        "parameters.target_course_code: only if user asks about a UC target (e.g., 'CS 61A') for equivalency.\n"
        "parameters.target_institution: the UC campus name if mentioned (e.g., 'UC Berkeley').\n"
        "filters.focus_only: one of 'cs','math','science','all', or null.\n"
        "filters.required_only: boolean.\n"
        "filters.domains_completed: list among 'cs','math','science'.\n"
        "filters.completed_courses: array of normalized DVC course codes (DEPT-NUM) if the user lists them.\n"
        "filters.categories: array of category names/phrases the user requests (e.g., 'major preparation','breadth','general education'). "
        "Use the user's wording; do not invent categories.\n"
        "If unsure, return null or empty arrays rather than guessing."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            temperature=0
        )
        data = json.loads(resp.choices[0].message.content)

        #defaults
        data.setdefault("intent", "find_requirements")
        data.setdefault("parameters", {})
        data.setdefault("filters", {})
        params = data["parameters"]
        filt = data["filters"]

        #campuses (array + single)
        campuses_raw = params.get("campuses", [])
        if not isinstance(campuses_raw, list):
            campuses_raw = []
        single_campus = params.get("campus")
        if isinstance(single_campus, str) and single_campus.strip():
            campuses_raw.append(single_campus)
        campuses_raw.extend(detect_campuses_from_query(user_message))

        campuses_norm: List[str] = []
        for c in campuses_raw:
            if not isinstance(c, str):
                continue
            det = detect_campus_from_query(c) or c.upper().strip()
            if det in PRETTY_CAMPUS:
                campuses_norm.append(det)
        campuses_norm = sorted(set(campuses_norm))
        params["campus"] = campuses_norm[0] if campuses_norm else None
        params["campuses"] = campuses_norm

        #filters: focus / required
        focus = filt.get("focus_only")
        if isinstance(focus, str):
            focus = focus.lower().strip()
            if focus not in {"cs", "math", "science", "all"}:
                focus = None
        else:
            focus = None
        filt["focus_only"] = focus
        filt["required_only"] = bool(filt.get("required_only", False))

        #domains_completed
        domains = filt.get("domains_completed") or []
        if isinstance(domains, list):
            domains = {d for d in (x.lower().strip() for x in domains) if d in {"cs","math","science"}}
        else:
            domains = set()
        filt["domains_completed"] = sorted(domains)

        #completed_courses
        comp = filt.get("completed_courses") or []
        if isinstance(comp, list):
            norm = {_normalize_single_code(x) for x in comp if isinstance(x, str)}
        else:
            norm = set()
        norm |= parse_completed_freeform(user_message)
        filt["completed_courses"] = sorted(norm)

        #new: categories (LLM + local merge)
        cats = filt.get("categories") or []
        cats = cats if isinstance(cats, list) else []
        cats_local = normalize_categories_freeform(user_message)
        merged_cats = sorted(set([c for c in cats if isinstance(c, str) and c.strip()] + cats_local))
        filt["categories"] = merged_cats

        return data
    except Exception:
        return {
            "intent": "find_requirements",
            "parameters": {"campus": None, "campuses": [], "target_course_code": None, "target_institution": None},
            "filters": {"focus_only": None, "required_only": False, "domains_completed": [], "completed_courses": [], "categories": []}
        }

#LLM format
def llm_format_response(client: OpenAI,
                        campus_key: str,
                        rows: List[Dict[str, Any]],
                        parsed: Dict[str, Any],
                        completed_courses: Set[str],
                        completed_domains: Set[str],
                        plain: bool = False) -> str:
    campus_name = PRETTY_CAMPUS.get(campus_key, campus_key)
    items = []
    for r in rows:
        items.append({
            "course": (r.get("dvc_code") or "").strip(),
            "title": (r.get("dvc_title") or "").strip(),
            "units": r.get("dvc_units", "")
        })

    if plain:
        if not items:
            return f"Transfer prep for {campus_name}:\nNo DVC course mappings found."
        lines = [f"Transfer prep for {campus_name}:"]
        if completed_domains or completed_courses:
            lines.append(f"(excluding completed domains: {', '.join(sorted(completed_domains)) or 'none'}; "
                         f"completed courses: {', '.join(sorted(completed_courses)) or 'none'})")
        for it in items:
            parts = [p for p in [
                it.get("course"),
                it.get("title"),
                (f"{it.get('units')} units" if (it.get("units") not in (None, "")) else None)
            ] if p]
            lines.append("• " + " — ".join(parts))
        return "\n".join(lines)

    payload = {
        "campus": campus_name,
        "intent": parsed.get("intent"),
        "parameters": parsed.get("parameters", {}),
        "filters": parsed.get("filters", {}),
        "excluding": {
            "completed_domains": sorted(list(completed_domains)),
            "completed_courses": sorted(list(completed_courses)),
        },
        "courses": items
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                 "Format UC transfer mappings only (no availability/schedule). "
                 "Output:\n"
                 "• One summary line: 'Transfer prep for <Campus>:'\n"
                 "• Optionally: parenthetical note like '(excluding completed domains: science; completed courses: COMSC-110)'\n"
                 "• Bullets: '• COMSC-200 — Object Oriented Programming C++ (4 units)'\n"
                 "If empty, say: 'No DVC course mappings found.'"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0.2
        )
        text = resp.choices[0].message.content.strip()
        if text:
            return text
    except Exception:
        pass

    #deterministic fallback
    if not items:
        return f"Transfer prep for {campus_name}:\nNo DVC course mappings found."
    lines = [f"Transfer prep for {campus_name}:"]
    if completed_domains or completed_courses:
        lines.append(f"(excluding completed domains: {', '.join(sorted(completed_domains)) or 'none'}; "
                     f"completed courses: {', '.join(sorted(completed_courses)) or 'none'})")
    for it in items:
        parts = [p for p in [
            it.get("course"),
            it.get("title"),
            (f"{it.get('units')} units" if (it.get('units') not in (None, "")) else None)
        ] if p]
        lines.append("• " + " — ".join(parts))
    return "\n".join(lines)

#new: multi-campus formatter wrapper
def llm_format_response_multi(client: OpenAI,
                              campus_keys: List[str],
                              campus_to_rows: Dict[str, List[Dict[str, Any]]],
                              parsed: Dict[str, Any],
                              completed_courses: Set[str],
                              completed_domains: Set[str],
                              plain: bool = False) -> str:
    chunks = []
    for ck in campus_keys:
        rows = campus_to_rows.get(ck, [])
        chunks.append(
            llm_format_response(client, ck, rows, parsed, completed_courses, completed_domains, plain=plain)
        )
    return "\n\n".join(chunks)

#logging
LOG_CSV = "data/conversation_log.csv"
LOG_JSONL = "data/user_log.jsonl"
SESSION_ID = str(uuid.uuid4())

def ensure_log_dir():
    for p in [LOG_CSV, LOG_JSONL]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

def append_logs(user_prompt: str,
                parsed: Dict[str, Any],
                response: str,
                campus_key: str,
                rows: List[Dict[str, Any]],
                completed_courses: Set[str],
                completed_domains: Set[str],
                query_id: int):
    ensure_log_dir()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "session_id": SESSION_ID,
        "query_id": query_id,
        "timestamp": ts,
        "campus": PRETTY_CAMPUS.get(campus_key, campus_key),
        "prompt": user_prompt,
        "parsed_json": json.dumps(parsed, ensure_ascii=False),
        "completed_domains": ", ".join(sorted(completed_domains)) if completed_domains else "",
        "completed_courses": ", ".join(sorted(completed_courses)) if completed_courses else "",
        "results": len(rows),
        "response": (response or "").replace("\n", "\\n"),
    }
    new_file = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

#UI helpers
def print_lists(campus_key: str,
                remaining_rows: List[Dict[str, Any]],
                completed_courses: Set[str],
                completed_domains: Set[str]):
    campus_name = PRETTY_CAMPUS.get(campus_key, campus_key)
    excl = f"(excluding: domains={', '.join(sorted(completed_domains)) or 'none'}; " \
           f"courses={', '.join(sorted(completed_courses)) or 'none'})"
    print(f"\nRemaining courses for {campus_name} {excl}:\n")
    for c in remaining_rows:
        code = (c.get("dvc_code") or "").strip()
        title = (c.get("dvc_title") or "").strip()
        units = c.get("dvc_units", "")
        parts = [code] if code else []
        if title: parts.append(title)
        if units != "":
            parts.append(f"{units}" if "unit" in str(units).lower() else f"{units} units")
        print("- " + " — ".join(parts))

def parse_cli_campuses(opt_str: Optional[str]) -> List[str]:
    if not opt_str:
        return []
    raw = [s.strip() for s in opt_str.split(",") if s.strip()]
    out: List[str] = []
    for r in raw:
        det = detect_campus_from_query(r) or r.upper()
        if det in PRETTY_CAMPUS:
            out.append(det)
    return sorted(set(out))

def interactive_session(client: OpenAI, data: Dict[str, Any], args) -> None:
    query_id = 1
    print("\nAsk a transfer question (e.g., 'what do I need for uc berkeley cs', "
          "'uc davis & ucsd cs requirements', or 'major preparation only for berkeley and davis').")
    try:
        user_q = input("> ").strip()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting session.")
        return
    if not user_q:
        user_q = "major preparation only for uc berkeley cs and uc davis"

    parsed = llm_parse_user_message(client, user_q)
    if args.json_only:
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        campus_keys_for_log = parsed.get("parameters", {}).get("campuses") or list(data.keys())
        campus_key_for_log = (campus_keys_for_log[0] if campus_keys_for_log else (next(iter(data.keys())) if data else "UCB"))
        append_logs(user_q, parsed, "", campus_key_for_log, [], set(), set(), query_id)
        return

    #campuses for this session
    campus_keys = []
    if args.campuses:
        campus_keys = [ck for ck in args.campuses if ck in data]
    if not campus_keys:
        campus_keys = parsed.get("parameters", {}).get("campuses") or detect_campuses_from_query(user_q)
    if not campus_keys:
        print("Sorry, I couldn't detect a campus. Try UC Berkeley (UCB), UC Davis (UCD), or UC San Diego (UCSD).")
        return
    campus_keys = [ck for ck in campus_keys if ck in data]
    if not campus_keys:
        print("Could not find data for the requested campus(es).")
        return

    #persistent state across the session
    seed_prefs = parse_preferences_seed(user_q)
    completed_courses: Set[str] = set(parsed["filters"]["completed_courses"])
    completed_domains: Set[str] = set(parsed["filters"]["domains_completed"])
    focus_only = parsed["filters"]["focus_only"]
    required_only = parsed["filters"]["required_only"]
    categories_only: List[str] = parsed["filters"].get("categories") or seed_prefs.get("seed_categories") or []

    campus_to_all_rows: Dict[str, List[Dict[str, Any]]] = {ck: collect_course_rows(data[ck]) for ck in campus_keys}

    def recompute_remaining() -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for ck in campus_keys:
            rows = campus_to_all_rows.get(ck, [])
            out[ck] = filter_rows(rows, seed_prefs, completed_courses, completed_domains,
                                  focus_only, required_only, categories_only=categories_only)
        return out

    campus_to_remaining = recompute_remaining()

    #initial print
    for ck in campus_keys:
        print(f"\nFound {len(campus_to_remaining[ck])} mapped DVC courses for {PRETTY_CAMPUS[ck]}.\n")
        print_lists(ck, campus_to_remaining[ck], completed_courses, completed_domains)

    remove_patterns = [
        r"\bremove\s+(?P<campus>.+)",
        r"\bdrop\s+(?P<campus>.+)",
        r"\bexclude\s+(?P<campus>.+)",
    ]

    while True:
        print("\nYou can:")
        print("  • Paste completed codes (e.g., 'COMSC-110, MATH-192') or say 'I finished all my science'")
        print("  • Focus: 'cs only', 'math only', 'science only', 'all courses'")
        print("  • Required: 'required only' / 'show all'")
        print("  • Category: 'category:\"major preparation\"', 'only breadth', 'show general education only'")
        print("  • Add campuses: 'also add uc san diego', 'include berkeley and davis'")
        print("  • Remove a campus: 'remove uc davis', 'drop berkeley'")
        print("  • Clear categories: 'clear categories'   • Reset completions: 'reset'")
        print("  • Finish: 'done'")
        try:
            follow = input("> ").strip()
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting session.")
            return
        if not follow:
            continue

        low = normalize_typos(follow)
        if low == "done":
            formatted = llm_format_response_multi(
                client,
                campus_keys,
                campus_to_remaining,
                parsed,
                completed_courses,
                completed_domains,
                plain=args.plain
            )
            print("\n---")
            if args.demo:
                print("BEFORE (parser JSON):")
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
                print("\nAFTER (formatted):")
            else:
                print("Final summary (combined):\n")
            print(formatted)
            append_logs(
                user_q,
                parsed,
                formatted,
                "MULTI:" + ",".join(campus_keys),
                [r for ck in campus_keys for r in campus_to_remaining.get(ck, [])],
                completed_courses,
                completed_domains,
                query_id
            )
            if args.demo:
                print(f"\nLogged → {LOG_CSV} and {LOG_JSONL}")
            break

        if low == "reset":
            completed_courses.clear()
            completed_domains.clear()
            campus_to_remaining = recompute_remaining()
            print("\nCompleted cleared.")
            for ck in campus_keys:
                print_lists(ck, campus_to_remaining[ck], completed_courses, completed_domains)
            continue

        if "clear categories" in low:
            categories_only = []
            campus_to_remaining = recompute_remaining()
            print("\nCategories cleared.")
            for ck in campus_keys:
                print_lists(ck, campus_to_remaining[ck], completed_courses, completed_domains)
            continue

        #quick path: explicit remove campus
        removed_any = False
        for pat in remove_patterns:
            m = re.search(pat, low)
            if m:
                cand = m.group("campus")
                cand_keys = detect_campuses_from_query(cand)
                for ck in list(campus_keys):
                    if ck in cand_keys:
                        campus_keys.remove(ck)
                        removed_any = True
                if removed_any:
                    campus_to_remaining = {ck: campus_to_remaining.get(ck, []) for ck in campus_keys}
                    print("\nRemaining campuses:", ", ".join(PRETTY_CAMPUS[ck] for ck in campus_keys) or "none")
                    if not campus_keys:
                        print("No campuses selected; type 'include <campus>' to add one back (e.g., 'include uc berkeley').")
                    else:
                        campus_to_remaining = recompute_remaining()
                        for ck in campus_keys:
                            print_lists(ck, campus_to_remaining[ck], completed_courses, completed_domains)
                break
        if removed_any:
            continue

        #re-parse follow-up for filters + campuses + categories
        query_id += 1
        follow_parsed = llm_parse_user_message(client, follow)

        #merge completed state
        completed_courses |= set(follow_parsed["filters"]["completed_courses"]) | parse_completed_freeform(follow)
        completed_domains |= set(follow_parsed["filters"]["domains_completed"])

        #update focus/required if specified
        if follow_parsed["filters"]["focus_only"] is not None:
            focus_only = follow_parsed["filters"]["focus_only"]
        if isinstance(follow_parsed["filters"]["required_only"], bool):
            required_only = follow_parsed["filters"]["required_only"]

        #update categories 
        cats_from_llm = follow_parsed["filters"].get("categories") or []
        cats_local = normalize_categories_freeform(follow)
        new_cats = sorted(set((categories_only or []) + cats_from_llm + cats_local))
        if new_cats != categories_only:
            categories_only = new_cats
            if categories_only:
                print("Active categories:", ", ".join(categories_only))

        #detect/merge campuses 
        new_camps = follow_parsed.get("parameters", {}).get("campuses") or detect_campuses_from_query(follow)
        added = False
        for ck in new_camps:
            if ck in data and ck not in campus_keys:
                campus_keys.append(ck)
                campus_to_all_rows[ck] = collect_course_rows(data[ck])
                added = True
        if added:
            print("\nIncluded campuses:", ", ".join(PRETTY_CAMPUS[ck] for ck in campus_keys))

        #recompute & show
        campus_to_remaining = recompute_remaining()
        print("\nUpdated results:")
        for ck in campus_keys:
            print_lists(ck, campus_to_remaining[ck], completed_courses, completed_domains)

#start
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is missing. Put it in your .env file.")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Show BEFORE (parser JSON) → AFTER (formatted) and log paths.")
    parser.add_argument("--plain", action="store_true", help="Bypass LLM formatter and use deterministic bullets.")
    parser.add_argument("--json-only", action="store_true", help="Only print parser JSON and exit.")
    parser.add_argument("--campuses", type=str, help="Comma-separated campuses to include (e.g., 'UCB,UCD' or 'berkeley,ucsd').")
    args = parser.parse_args()

    client = OpenAI(api_key=api_key)

    data = load_all_data([
        os.path.join("data", "uc*.json"),
        os.path.join("agreements_25-26", "*.json"),
    ])
    print("✅ Loaded campuses:", sorted(list(data.keys())))
    print(f"CSV log → {LOG_CSV}")
    print(f"JSONL log → {LOG_JSONL}")
    if not data:
        print("⚠️ No campus files loaded. Check data/ and agreements_25-26/")
        return

    #normalize CLI campuses now
    if args.campuses:
        args.campuses = parse_cli_campuses(args.campuses)
        if args.campuses:
            print("CLI campuses →", ", ".join(PRETTY_CAMPUS[c] for c in args.campuses))
        else:
            print("CLI campuses → none recognized; falling back to parsed input")

    while True:
        try:
            interactive_session(client, data, args)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        try:
            print("\nStart another query session? (yes/no)")
            again = input("> ").strip().lower()
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        if again not in ("y", "yes"):
            print("Goodbye! 👋")
            break

if __name__ == "__main__":
    main()
