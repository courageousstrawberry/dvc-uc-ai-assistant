import os
import json
import glob
import re
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dotenv import load_dotenv
from openai import OpenAI

#campus names
CAMPUS_ALIASES = {
    "UCB": ["uc berkeley", "berkeley", "ucb"],
    "UCD": ["uc davis", "davis", "ucd"],
    "UCI": ["uc irvine", "irvine", "uci"],
    "UCLA": ["ucla", "uc los angeles", "los angeles"],
    "UCSD": ["uc san diego", "san diego", "ucsd"],
}
PRETTY_CAMPUS = {
    "UCB": "UC Berkeley",
    "UCD": "UC Davis",
    "UCI": "UC Irvine",
    "UCLA": "UCLA",
    "UCSD": "UC San Diego",
}

#common typos
TYPO_FIXES = {
    r"\busb\b": "uc berkeley",
    r"\bucb\b": "uc berkeley",
    r"\bberkley\b": "berkeley",
    r"\bucsd\b": "uc san diego",
    r"\buc sd\b": "uc san diego",
    r"\buc la\b": "ucla",
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

#intent
def parse_preferences(q: str) -> dict:
    t = " " + q.lower().strip() + " " 

    want_cs = any(x in t for x in [" cs ", " cs?", "cs,", "cs.", "comsc", "computer science", "programming", "data structures"])
    want_math = any(x in t for x in [" math ", "calculus", "linear algebra", "differential equations"])
    want_science = any(x in t for x in [" science ", " physics", " chemistry", " biology", " bio ", " chem ", " phys "])

    #exclusive mode: if a single domain is asked, restrict to that domain only
    exclusive_domain = None
    if want_cs and not (want_math or want_science):
        exclusive_domain = "cs"
    elif want_math and not (want_cs or want_science):
        exclusive_domain = "math"
    elif want_science and not (want_cs or want_math):
        exclusive_domain = "science"

    return {
        "required_only": any(x in t for x in ["required only", "only required", "must have", "need all"]),
        "want_math": want_math,
        "want_cs": want_cs,
        "want_science": want_science,
        "exclusive_domain": exclusive_domain,  
    }

#domain checks (code-first, then category/title)
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

#code normalization & completed parsing
def _normalize_single_code(raw: str) -> str:
    s = raw.upper().strip()
    s = s.replace(" ", "-")
    #common synonyms -> DVC prefix
    if s.startswith(("CS-", "COMPSCI-", "COMSCI-", "COMPSC-")):
        s = "COMSC-" + s.split("-", 1)[1]
    #make sure: hyphen between dept and number (e.g., COMSC110 -> COMSC-110)
    m = re.match(r"^([A-Z&]+)[- ]?(\d+[A-Z]?)$", s)
    if m:
        s = f"{m.group(1)}-{m.group(2)}"
    return s

def _split_multi_code_field(code_field: str) -> List[str]:
    #ex: "COMSC-110/165/200/210" -> ["COMSC-110","COMSC-165","COMSC-200","COMSC-210"]
    if not code_field:
        return []
    code_field = code_field.strip().upper()
    parts = re.split(r"[\/,&]| and ", code_field, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    expanded: List[str] = []
    prefix = None
    for p in parts:
        if "-" in p:
            prefix = p.split("-", 1)[0]
            expanded.append(_normalize_single_code(p))
        else:
            if prefix:
                expanded.append(_normalize_single_code(f"{prefix}-{p}"))
            else:
                expanded.append(_normalize_single_code(p))
    return expanded

def parse_completed_freeform(text: str) -> Set[str]:
    """
    Parse a freeform line like:
      'COMSC-110, math 192 & phys 230'
    into normalized codes: {'COMSC-110', 'MATH-192', 'PHYS-230'}
    """
    tokens: Set[str] = set()
    for code in re.findall(r"\b([A-Za-z]{2,}[- ]?\d+[A-Za-z]?)\b", text, flags=re.IGNORECASE):
        dep = code.strip().split()[0].upper()
        if any(dep.startswith(pfx) for pfx in
               ["COMSC", "CS", "COMPSCI", "COMSCI", "COMPSC", "MATH", "PHYS", "CHEM", "BIOSC", "BIOL", "ENGIN", "ENGL"]):
            tokens.add(_normalize_single_code(code))
    return tokens

#data load
def load_all_data(paths: List[str]) -> Dict[str, Any]:
    """
    Load campus JSONs from any number of glob paths.
    Campus key inferred from filename prefix before first underscore: ucb_ -> UCB, etc.
    """
    all_data: Dict[str, Any] = {}
    for pattern in paths:
        for path in glob.glob(pattern):
            base = os.path.basename(path)
            campus_key = base.split("_")[0].upper()  # ucb_..., ucd_..., uci_..., ucla_..., ucsd_...
            try:
                with open(path, "r", encoding="utf-8") as f:
                    all_data[campus_key] = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    return all_data

#JSON traversal
def collect_course_rows(campus_json: Any) -> List[Dict[str, Any]]:
    """
    Traverse the campus JSON and collect rows with:
      - category (string)
      - minimum_required (raw value)
      - dvc_code/title/units (flattened; DVC may be dict or list)
    """
    out: List[Dict[str, Any]] = []

    def _recurse(obj: Any):
        if isinstance(obj, dict):
            if "Category" in obj and "Courses" in obj:
                category = obj.get("Category", "")
                minimum_required = obj.get("Minimum_Required", "")
                courses = obj.get("Courses", [])
                if isinstance(courses, list):
                    for pair in courses:
                        dvc_block = pair.get("DVC")
                        if dvc_block is None:
                            continue
                        dvc_items = dvc_block if isinstance(dvc_block, list) else [dvc_block]
                        for d in dvc_items:
                            if not isinstance(d, dict):
                                continue
                            out.append({
                                "category": category,
                                "minimum_required": minimum_required,
                                "dvc_code": d.get("Course_Code", "") or d.get("Code", ""),
                                "dvc_title": d.get("Title", ""),
                                "dvc_units": d.get("Units", "") or d.get("units", ""),
                            })
            for v in obj.values():
                _recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                _recurse(item)

    _recurse(campus_json)

    #deduplicate by code while preserving the order
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for row in out:
        key = (row.get("dvc_code") or "").strip()
        if key and key not in seen:
            deduped.append(row)
            seen.add(key)
    return deduped

#filtering: exclusive domain support
def filter_rows(rows: List[Dict[str, Any]], prefs: dict, completed: Set[str]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    completed_upper = {c.upper() for c in completed}

    #exclusive domain: keep only that domain.
    exclusive = prefs.get("exclusive_domain")

    for r in rows:
        #domain gating: exclusive
        if exclusive == "cs" and not is_cs_row(r):
            continue
        if exclusive == "math" and not is_math_row(r):
            continue
        if exclusive == "science" and not is_science_row(r):
            continue

        #non-exclusive: allow any that match at least one of the requested domain
        if exclusive is None:
            want_any = any([prefs.get("want_cs"), prefs.get("want_math"), prefs.get("want_science")])
            if want_any:
                ok = False
                if prefs.get("want_cs"):
                    ok = ok or is_cs_row(r)
                if prefs.get("want_math"):
                    ok = ok or is_math_row(r)
                if prefs.get("want_science"):
                    ok = ok or is_science_row(r)
                if not ok:
                    continue
            # If no domain requested at all, show everything.

        #required only filter
        if prefs.get("required_only"):
            mr = str(r.get("minimum_required", "")).lower()
            if not (mr == "all" or (mr.isdigit() and int(mr) > 0)):
                continue

        #completed filter: any dvc code
        code_field = r.get("dvc_code", "")
        codes_in_row = set(_split_multi_code_field(code_field))
        if codes_in_row & completed_upper:
            continue

        filtered.append(r)

    return filtered

#formatting
def format_requirements_with_llm(client: OpenAI, campus_key: str, courses: List[Dict[str, Any]]) -> str:
    campus_name = PRETTY_CAMPUS.get(campus_key, campus_key)
    if not courses:
        return f"No DVC course mappings found for {campus_name} CS."

    lines = []
    for c in courses:
        code = c.get("dvc_code", "").strip()
        title = c.get("dvc_title", "").strip()
        units = c.get("dvc_units", "")
        parts = [code] if code else []
        if title:
            parts.append(title)
        if units != "":
            if isinstance(units, (int, float)):
                parts.append(f"{units} units")
            else:
                parts.append(f"{units}" if "unit" in str(units).lower() else f"{units} units")
        if parts:
            lines.append(" — ".join(parts))
    context = "\n".join(lines)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise DVC→UC CS transfer advisor."},
                {"role": "user", "content": f"List the DVC courses needed to transfer for CS at {campus_name}."},
                {"role": "assistant", "content": f"Here are parsed DVC courses:\n{context}\n\nFormat as bullet points: CODE — Title (Units)."}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "\n".join([f"• {line}" for line in lines]) or f"No DVC course mappings found for {campus_name} CS."

def print_lists(campus_key: str, remaining_rows: List[Dict[str, Any]], completed: Set[str]):
    campus_name = PRETTY_CAMPUS.get(campus_key, campus_key)
    print(f"\nRemaining courses for {campus_name} (excluding completed: {', '.join(sorted(completed)) if completed else 'none'}):\n")
    for c in remaining_rows:
        code = (c.get("dvc_code") or "").strip()
        title = (c.get("dvc_title") or "").strip()
        units = c.get("dvc_units", "")
        parts = [code] if code else []
        if title:
            parts.append(title)
        if units != "":
            parts.append(f"{units}" if "unit" in str(units).lower() else f"{units} units")
        print("- " + " — ".join(parts))

#.csv
LOG_FILE = os.path.join("data", "conversation_log.csv")

def ensure_log_dir():
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

def append_to_log(user_input: str,
                  response: str,
                  campus_key: str,
                  prefs: dict,
                  completed: Set[str],
                  results_count: int):
    """Append a timestamped record to CSV."""
    ensure_log_dir()
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "campus": PRETTY_CAMPUS.get(campus_key, campus_key),
        "user_input": user_input,
        "exclusive_domain": prefs.get("exclusive_domain") or "",
        "want_cs": str(bool(prefs.get("want_cs"))),
        "want_math": str(bool(prefs.get("want_math"))),
        "want_science": str(bool(prefs.get("want_science"))),
        "required_only": str(bool(prefs.get("required_only"))),
        "completed": ", ".join(sorted(completed)) if completed else "",
        "results_count": str(results_count),
        "response": response.replace("\n", "\\n"),
    }
    is_new = not os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"⚠️ Could not write to log: {e}")

#conversational flow
def answer_one_question(client: OpenAI, data: Dict[str, Any]) -> None:
    """Handle one question + interactive follow-ups; allow mid-loop new questions or filter tweaks."""
    print("\nAsk a question (e.g., 'cs for ucb', 'math only for ucla cs', or 'ucsd cs requirements'):")
    user_q = input("> ").strip()
    if not user_q:
        user_q = "What DVC courses do I need for UC Irvine CS?"

    #initial state
    campus_key = detect_campus_from_query(user_q)
    if not campus_key:
        print("Sorry, I couldn't detect a UC campus. Try UC Berkeley, UC Davis, UC Irvine, UCLA, or UC San Diego.")
        return
    campus_json = data.get(campus_key)
    if not campus_json:
        print(f"Could not find data for {campus_key}.")
        return

    prefs = parse_preferences(user_q)
    all_rows = collect_course_rows(campus_json)
    completed: Set[str] = set()

    #initial display
    remaining = filter_rows(all_rows, prefs, completed)
    print(f"\nParsed {len(remaining)} DVC course rows for {PRETTY_CAMPUS.get(campus_key, campus_key)} (after filters).")
    print_lists(campus_key, remaining, completed)

    #conversational loop
    while True:
        print("\nYou can:")
        print(" - type course codes you’ve completed (e.g., 'COMSC-110, MATH-192')")
        print(" - change filters (e.g., 'math only', 'required only', 'cs only', 'science only')")
        print(" - ask a NEW question for a different campus (e.g., 'what do I need for uc berkeley cs?')")
        print(" - 'reset' to clear completed, or 'done' to finish")
        follow = input("> ").strip()
        if not follow:
            #ends the loop
            break

        low = normalize_typos(follow)

        #commands
        if low == "done":
            break
        if low == "reset":
            completed.clear()
            remaining = filter_rows(all_rows, prefs, completed)
            print("\nCompleted list cleared.")
            print_lists(campus_key, remaining, completed)
            continue

        #1 completed courses
        newly_completed = parse_completed_freeform(follow)
        if newly_completed:
            pretty = ", ".join(sorted(newly_completed))
            print(f"Noted completed: {pretty}")
            completed |= newly_completed
            remaining = filter_rows(all_rows, prefs, completed)
            print_lists(campus_key, remaining, completed)
            continue

        #2 new question (y/n)
        new_campus = detect_campus_from_query(follow)
        if new_campus:
            campus_key = new_campus
            campus_json = data.get(campus_key)
            if not campus_json:
                print(f"Could not find data for {campus_key}.")
                continue
            prefs = parse_preferences(follow)
            all_rows = collect_course_rows(campus_json)
            completed.clear()
            remaining = filter_rows(all_rows, prefs, completed)
            print(f"\nSwitched to {PRETTY_CAMPUS.get(campus_key, campus_key)}.")
            print_lists(campus_key, remaining, completed)
            continue

        #3 filter out any tweaks
        maybe_new_prefs = parse_preferences(follow)
        if any([maybe_new_prefs.get("exclusive_domain"),
                maybe_new_prefs.get("want_cs"),
                maybe_new_prefs.get("want_math"),
                maybe_new_prefs.get("want_science"),
                maybe_new_prefs.get("required_only")]):
            prefs = maybe_new_prefs
            remaining = filter_rows(all_rows, prefs, completed)
            print("\nUpdated filters.")
            print_lists(campus_key, remaining, completed)
            continue

        #4 QUESTION-LIKE text but no campus keyword
        if any(w in low for w in ["what", "need", "courses", "requirements", "transfer", "show"]):
            prefs = parse_preferences(follow)
            remaining = filter_rows(all_rows, prefs, completed)
            print("\nUpdated based on your question for the same campus.")
            print_lists(campus_key, remaining, completed)
            continue

       
        print("I didn't detect any course codes or campus/filter change. Try 'COMSC-110', 'math only', or 'uc berkeley cs'.")

    #final formatted summary
    print("\n---\nFinal summary:\n")
    formatted = format_requirements_with_llm(client, campus_key, remaining)
    print(formatted)

    #log the session to CSV
    try:
        append_to_log(
            user_input=user_q,
            response=formatted,
            campus_key=campus_key,
            prefs=prefs,
            completed=completed,
            results_count=len(remaining)
        )
    except Exception as e:
        print(f"⚠️ Logging error: {e}")

def main():
    #1 API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is missing. Put it in your .env file.")
        return

    #2 Load data from both locations
    data = load_all_data([
        os.path.join("data", "uc*.json"),
        os.path.join("agreements_25-26", "*.json"),
    ])
    print("✅ Loaded campuses:", sorted(list(data.keys())))
    if not data:
        print("⚠️ No campus files loaded. Check data/ and agreements_25-26/")
        return

    #3 ping OpenAI once
    client = OpenAI(api_key=api_key)
    try:
        ping = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a DVC course advisor assistant."},
                {"role": "user", "content": "Say: API connection successful."}
            ],
            temperature=0.2
        )
        print(ping.choices[0].message.content.strip())
    except Exception as e:
        print("API call failed:", e)
        return

    #4 handle one Q&A session
    answer_one_question(client, data)

if __name__ == "__main__":
    while True:
        main()
        print("\nWould you like to ask another question? (yes/no)")
        again = input("> ").strip().lower()
        if again not in ("y", "yes"):
            print("Goodbye! 👋")
            break
