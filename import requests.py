import re
import unicodedata
import requests
import pandas as pd
import time

# Optional fuzzy matching (graceful fallback)
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

# ------------------------------
# Categories + acronyms/short forms
# ------------------------------
CATEGORIES = {
    "Reminiscence therapy & variants": [
        "reminiscence therapy",
        "rt",
        "life review therapy",
        "lrt",
        "memory recall therapy",
        "life story work",
        "lsw",
        "narrative therapy",
        "nt",
        "life history approach",
        "storytelling therapy",
        "group reminiscence",
        "life story",
    ],
    "Dementia-related": [
        "dementia",
        "alzheimers",
        "alzheimer's disease",
        "ad",
        "mild cognitive impairment",
        "mci",
        "neurocognitive disorder",
        "ncd",
        "vascular dementia",
        "vd",
        "frontotemporal dementia",
        "ftd",
        "lewy body dementia",
        "lbd",
        "bpsd",
    ],
    "Non-pharmacological approaches": [
        "non-pharmacological",
        "nonpharmacological",
        "npi",
        "behavioral intervention",
        "psychosocial intervention",
        "psi",
        "lifestyle intervention",
        "therapeutic activity",
        "ta",
        "occupational therapy",
        "ot",
        "music therapy",
        "mt",
        "art therapy",
        "at",
        "pet therapy",
        "animal-assisted therapy",
        "aat",
        "exercise therapy",
        "et",
    ],
    "Memory & cognition interventions": [
        "memory intervention",
        "mi",
        "memory training",
        "mt",
        "memory therapy",
        "cognitive training",
        "ct",
        "cognitive stimulation",
        "cs",
        "cognitive rehabilitation",
        "cr",
        "cognitive therapy",
        "brain training",
        "bt",
        "mental stimulation",
        "ms",
        "neurocognitive training",
        "neuropsychological rehabilitation",
        "npr",
    ],
    "Group / social engagement": [
        "social engagement therapy",
        "peer group therapy",
        "community-based therapy",
        "cbt",  # note: also Cognitive Behavioral Therapy in other contexts
        "group therapy",
        "group-based",
        "social participation",
        "social engagement",
        "set",
    ],
    "Adjacent techniques & related terms": [
        "reality orientation therapy",
        "rot",
        "validation therapy",
        "vt",
        "biographical approach",
        "recollection therapy",
    ],
}

# Union of all keywords for quick checks
ALL_KEYWORDS = sorted({kw for kws in CATEGORIES.values() for kw in kws})

# ------------------------------
# API details
# ------------------------------
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/author/search"
PUBLICATION_FIELDS = "papers.title,papers.year,papers.abstract"

# ------------------------------
# Normalization & matching helpers
# ------------------------------

def normalize_text(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def keyword_to_regex(kw: str) -> re.Pattern:
    """
    Build a robust regex:
    - Acronyms/short forms: strict word boundaries, e.g., \bMCI\b (case-insensitive)
    - Phrases: allow spaces OR hyphens between words (e.g., 'non[-\s]?pharmacological')
    """
    kw_norm = kw.lower().strip()
    # treat something as acronym if it has no spaces and is <= 4 chars or all caps-ish
    is_acronym = (" " not in kw_norm) and (len(kw_norm) <= 4 or kw.upper() == kw)
    if is_acronym:
        pat = r"\b" + re.escape(kw_norm) + r"\b"
    else:
        # allow hyphen or one/more spaces between words
        parts = [re.escape(p) for p in kw_norm.split()]
        if len(parts) == 1:
            # single token but not acronym (e.g., 'alzheimers')
            pat = r"\b" + parts[0] + r"\b"
        else:
            pat = r"\b" + r"[\s\-]+".join(parts) + r"\b"
    return re.compile(pat, flags=re.IGNORECASE)

REGEX_CACHE = {kw: keyword_to_regex(kw) for kw in ALL_KEYWORDS}

def regex_matches(text: str, kw: str) -> bool:
    if not text:
        return False
    return REGEX_CACHE[kw].search(text) is not None

def fuzzy_match(text: str, kw: str, threshold: int = 88) -> bool:
    """Fuzzy partial match fallback (optional)."""
    if not RAPIDFUZZ_AVAILABLE or not text or not kw:
        return False
    try:
        # Use partial_ratio against normalized strings
        score = fuzz.partial_ratio(normalize_text(kw), normalize_text(text))
        return score >= threshold
    except Exception:
        return False

def find_matched_keywords(title: str, abstract: str) -> set:
    """
    Return set of keywords matched in title/abstract using:
    1) regex (primary)
    2) fuzzy fallback (optional)
    """
    text_title = title or ""
    text_abs = abstract or ""
    matches = set()
    for kw in ALL_KEYWORDS:
        if regex_matches(text_title, kw) or regex_matches(text_abs, kw) or fuzzy_match(text_title, kw) or fuzzy_match(text_abs, kw):
            matches.add(kw)
    return matches

def determine_match_type(title: str, abstract: str) -> str | None:
    """Title / Abstract / Both based on any keyword from ALL_KEYWORDS."""
    title_hit = any(regex_matches(title, kw) or fuzzy_match(title, kw) for kw in ALL_KEYWORDS)
    abstract_hit = any(regex_matches(abstract, kw) or fuzzy_match(abstract, kw) for kw in ALL_KEYWORDS)
    if title_hit and abstract_hit:
        return "Both"
    if title_hit:
        return "Title"
    if abstract_hit:
        return "Abstract"
    return None

def determine_match_categories(matched_keywords: set) -> list:
    """Map matched keywords to their category names."""
    cats = []
    for cat, kws in CATEGORIES.items():
        if any(kw in matched_keywords for kw in kws):
            cats.append(cat)
    return cats

# ------------------------------
# Semantic Scholar helpers
# ------------------------------

def search_author(author_name):
    params = {
        "query": author_name,
        "limit": 5,
        "fields": "authorId,name,affiliations"
    }
    resp = requests.get(SEMANTIC_SCHOLAR_API, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("data"):
        top_author = data["data"][0]
        return top_author["authorId"], top_author["name"], top_author.get("affiliations", [])
    return None, None, None

def get_author_papers(author_id):
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
    params = {"fields": PUBLICATION_FIELDS, "limit": 1000}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        print(f"Error fetching papers for author {author_id}: {resp.text}")
        return []
    data = resp.json()
    return data.get("papers", [])

# ------------------------------
# Main
# ------------------------------

def main(professor_names):
    results = []
    for name in professor_names:
        print(f"Searching author: {name}")
        try:
            author_id, author_name, affiliations = search_author(name)
        except Exception as e:
            print(f"Author search failed for {name}: {e}")
            continue

        if not author_id:
            print(f"Author not found: {name}")
            continue
        print(f"Found author: {author_name} (ID: {author_id})")

        papers = get_author_papers(author_id)
        print(f"Found {len(papers)} papers for {author_name}")

        for paper in papers:
            title = paper.get("title", "") or ""
            abstract = paper.get("abstract", "") or ""

            # Determine basic match type first (fast path)
            match_type = determine_match_type(title, abstract)
            if not match_type:
                continue

            # Extract exact matched keywords and map to categories
            matched_kws = sorted(find_matched_keywords(title, abstract))
            match_categories = determine_match_categories(set(matched_kws))

            results.append({
                "professor_name": author_name,
                "affiliations": "; ".join(affiliations) if affiliations else "",
                "paper_title": title,
                "paper_year": paper.get("year", ""),
                "paper_abstract": abstract,
                "match_type": match_type,
                "match_categories": "; ".join(match_categories) if match_categories else "",
                "matched_keywords": "; ".join(matched_kws) if matched_kws else ""
            })

        time.sleep(3)  # avoid rate limits

    df = pd.DataFrame(results)
    out = "professors_relevant_papers_full_abstract_fuzzy_categories.csv"
    df.to_csv(out, index=False)
    print(f"Done! Results saved to {out}")

if __name__ == "__main__":
    professor_list = [
        "Basia Belza",
        "Donna Berry",
        "Eeeseung Byun",
        "Chieh Cheng",
        "Kristen Childress",
        "Barbara Cochrane",
        "Paula Cox-North",
        "Cynthia Dougherty",
        "Maya Elías",
        "Margaret Heitkemper",
        "Frances Lewis",
        "Jingyi Li",
        "Susan McCurry",
        "Janet Primomo",
        "Kerry Reding",
        "David Reyes",
        "Anita Souza",
        "Megan Streur",
        "Hsin-Yi Tang",
        "Hilaire Thompson",
        "Alexi Vasbinder",
        "Allison Webel",
        "Nancy Fugate Woods",
        "Weichao Yuwen",
        "Oleg Zaslavsky",
        "Brenda Zierler"
    ]
    main(professor_list)