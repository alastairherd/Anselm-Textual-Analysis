#!/usr/bin/env python3
"""
Biblical Allusion Detector for Anselm of Canterbury

Detects direct quotes, paraphrases, and thematic echoes in Anselm's
Latin philosophical works against the Latin Vulgate.

Two-pipeline hybrid:
  1. N-gram matching  — catches direct quotes; fast, no API cost
  2. LLM via OpenRouter — catches paraphrases and thematic echoes

Usage:
  python detect_allusions.py --skip-llm          # n-gram only (free)
  python detect_allusions.py                      # full run (needs API key)
  python detect_allusions.py --help
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


# ── Normalization ──────────────────────────────────────────────────────────────

def normalize_medieval(text: str) -> str:
    """
    Normalize medieval Latin orthography for string comparison.
    Anselm uses classical orthography (u/v not distinguished);
    apply v→u and j→i so both Vulgate and Anselm normalize identically.
    """
    text = text.lower()
    text = text.replace('v', 'u')
    text = text.replace('j', 'i')
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Vulgate Loading ────────────────────────────────────────────────────────────

def load_vulgate(path: str):
    """
    Load vulgate.json and return:
      - verse_lookup: {(book, ch_1idx, vs_1idx): original_text}
      - norm_verses:  [(book, ch_1idx, vs_1idx, normalized_text), ...]

    The JSON structure is: {book_name: [[verse, ...], [chapter2...], ...]}
    Chapter and verse lists are 0-indexed; we convert to 1-indexed for output.

    Handles two known data artifacts:
      - 1,405 empty verse slots → skipped
      - 696 verses with leading '[' artifact character → stripped
    """
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    verse_lookup: dict = {}
    norm_verses: list = []

    for book, chapters in data.items():
        for ch_idx, chapter in enumerate(chapters):
            for vs_idx, verse in enumerate(chapter):
                if not verse or not verse.strip():
                    continue
                text = verse.lstrip('[')
                key = (book, ch_idx + 1, vs_idx + 1)
                verse_lookup[key] = text
                norm_text = normalize_medieval(text)
                if norm_text:
                    norm_verses.append((book, ch_idx + 1, vs_idx + 1, norm_text))

    print(f"Loaded {len(verse_lookup):,} verses from {len(data)} books", file=sys.stderr)
    return verse_lookup, norm_verses


# ── Anselm File Parsing ────────────────────────────────────────────────────────

def strip_page_markers(text: str) -> str:
    """Remove /NNN/ scan-page markers."""
    return re.sub(r'/\d+/', '', text)


def parse_anselm_file(path: str) -> list:
    """
    Parse an Anselm file into a list of section dicts:
      {'num': int, 'title': str, 'text': str}

    Two formats:
      Monologion / Proslogion  → TOC + *** separator + numbered body chapters
      Pro_insipiente / Responsio → [N] section markers throughout
    """
    filename = Path(path).stem
    with open(path, encoding='utf-8') as f:
        content = f.read()
    content = strip_page_markers(content)

    if filename in ('Monologion', 'Proslogion'):
        return _parse_toc_body_format(content, filename)
    else:
        return _parse_bracket_format(content, filename)


def _parse_toc_body_format(content: str, filename: str) -> list:
    """
    Parse Monologion / Proslogion:
    Find the last *** separator line; everything after is the body.
    Split body on chapter markers: optional-indent NUMBER. TITLE
    """
    sep_matches = list(re.finditer(r'(?m)^\*+\s*$', content))
    if not sep_matches:
        sep_matches = list(re.finditer(r'(?m)^\*+', content))
    if not sep_matches:
        raise ValueError(f"No *** separator found in {filename}")

    body_start = sep_matches[-1].end()
    body = content[body_start:]

    # Match lines like: "  1.  QUOD SIT QUIDDAM..." or "1.\tEXCITATIO..."
    chapter_pat = re.compile(r'(?m)^[ \t]{0,8}(\d+)\.[ \t]+([^\n]+)')
    matches = list(chapter_pat.finditer(body))

    sections = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[text_start:text_end].strip()
        # Skip TOC remnants or empty sections (TOC lines have no substantial text)
        if len(text) < 50:
            continue
        sections.append({'num': num, 'title': title, 'text': text})

    # Deduplicate by chapter num — keep the entry with the most text
    by_num: dict = {}
    for s in sections:
        n = s['num']
        if n not in by_num or len(s['text']) > len(by_num[n]['text']):
            by_num[n] = s

    return [by_num[n] for n in sorted(by_num)]


def _parse_bracket_format(content: str, filename: str) -> list:
    """
    Parse Pro_insipiente / Responsio:
    Split on [N] markers that appear alone on a line.
    """
    pattern = re.compile(r'(?m)^\s*\[(\d+)\]\s*$')
    matches = list(pattern.finditer(content))

    sections = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[text_start:text_end].strip()
        sections.append({'num': num, 'title': f'Section {num}', 'text': text})

    return sections


# ── N-gram Pipeline ────────────────────────────────────────────────────────────

def build_ngram_index(norm_verses: list, n: int = 4, max_verses: int = 3) -> dict:
    """
    Build an n-gram index from normalized Vulgate verses.

    Filters out n-grams appearing in more than max_verses verses
    (these are common phrases that produce false positives).

    Returns: {gram_tuple: {(book, ch, vs), ...}}
    """
    gram_to_verses: dict = defaultdict(set)
    for book, ch, vs, norm_text in norm_verses:
        tokens = norm_text.split()
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            gram_to_verses[gram].add((book, ch, vs))

    filtered = {
        gram: verses
        for gram, verses in gram_to_verses.items()
        if len(verses) <= max_verses
    }

    total = len(gram_to_verses)
    kept = len(filtered)
    print(
        f"N-gram index: {kept:,} unique {n}-grams kept "
        f"(filtered {total - kept:,} common grams from {total:,} total)",
        file=sys.stderr,
    )
    return filtered


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two normalized strings."""
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def ngram_search(
    section_text: str,
    ngram_index: dict,
    verse_lookup: dict,
    n: int = 4,
    min_hits: int = 1,
) -> list:
    """
    Search for Vulgate verses whose n-grams appear in section_text.

    min_hits=2 means at least 2 distinct 4-grams match, i.e. roughly
    5 consecutive matching words (adjacent 4-grams share 3 tokens).

    Returns list of match dicts sorted by descending ngram_score.
    """
    norm_text = normalize_medieval(section_text)
    tokens = norm_text.split()
    original_words = section_text.split()

    verse_hits: dict = defaultdict(int)
    verse_spans: dict = defaultdict(list)
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        if gram in ngram_index:
            for verse_key in ngram_index[gram]:
                verse_hits[verse_key] += 1
                verse_spans[verse_key].append(i)

    results = []
    for verse_key, hits in verse_hits.items():
        if hits < min_hits:
            continue
        book, ch, vs = verse_key
        vulgate_text = verse_lookup.get(verse_key, '')
        norm_vulgate = normalize_medieval(vulgate_text)
        sim = jaccard_similarity(norm_text, norm_vulgate)

        spans = verse_spans.get(verse_key, [])
        if spans and original_words:
            tok_start = max(0, min(spans) - 3)
            tok_end   = min(len(original_words), max(spans) + n + 3)
            anselm_snippet = ' '.join(original_words[tok_start:tok_end])
        else:
            anselm_snippet = ' '.join(original_words[:30])

        results.append({
            'book': book,
            'chapter': ch,
            'verse': vs,
            'vulgate_text': vulgate_text,
            'ngram_score': hits,
            'similarity': sim,
            'method': 'ngram',
            'allusion_type': 'direct_quote',
            'verified': True,
            'llm_confidence': '',
            'llm_explanation': '',
            'anselm_text': anselm_snippet,
            'judge_reasoning': '',
        })

    results.sort(key=lambda x: x['ngram_score'], reverse=True)
    return results


# ── LLM Pipeline ──────────────────────────────────────────────────────────────

# Maps common abbreviations → full book names used in vulgate.json
BOOK_ALIASES: dict = {
    # Pentateuch
    'Gen': 'Genesis', 'Ex': 'Exodus', 'Exod': 'Exodus', 'Lev': 'Leviticus',
    'Num': 'Numbers', 'Deut': 'Deuteronomy', 'Dt': 'Deuteronomy',
    # Historical
    'Josh': 'Joshua', 'Judg': 'Judges', 'Ruth': 'Ruth',
    '1 Sam': '1 Samuel', '1Sam': '1 Samuel', '2 Sam': '2 Samuel', '2Sam': '2 Samuel',
    '1 Kgs': '1 Kings', '1Kgs': '1 Kings', '2 Kgs': '2 Kings', '2Kgs': '2 Kings',
    '1 Chr': '1 Chronicles', '1Chr': '1 Chronicles',
    '2 Chr': '2 Chronicles', '2Chr': '2 Chronicles',
    'Ezra': 'Ezra', 'Neh': 'Nehemiah', 'Esth': 'Esther',
    # Wisdom
    'Job': 'Job', 'Ps': 'Psalms', 'Psalm': 'Psalms', 'Pss': 'Psalms',
    'Prov': 'Proverbs', 'Eccl': 'Ecclesiastes', 'Qoh': 'Ecclesiastes',
    'Song': 'Song of Solomon', 'Sg': 'Song of Solomon', 'Cant': 'Song of Solomon',
    # Prophets
    'Isa': 'Isaiah', 'Jer': 'Jeremiah', 'Lam': 'Lamentations',
    'Ezek': 'Ezekiel', 'Ez': 'Ezekiel', 'Dan': 'Daniel',
    'Hos': 'Hosea', 'Joel': 'Joel', 'Amos': 'Amos', 'Obad': 'Obadiah',
    'Jonah': 'Jonah', 'Jon': 'Jonah', 'Mic': 'Micah', 'Nah': 'Nahum',
    'Hab': 'Habakkuk', 'Zeph': 'Zephaniah', 'Hag': 'Haggai',
    'Zech': 'Zechariah', 'Mal': 'Malachi',
    # Deuterocanonical
    'Tob': 'Tobit', 'Jdt': 'Judith',
    '1 Macc': '1 Maccabees', '2 Macc': '2 Maccabees',
    'Wis': 'Wisdom', 'Sir': 'Sirach', 'Bar': 'Baruch',
    # Gospels & Acts
    'Matt': 'Matthew', 'Mt': 'Matthew', 'Mk': 'Mark', 'Lk': 'Luke',
    'Jn': 'John', 'Acts': 'Acts',
    # Epistles
    'Rom': 'Romans',
    '1 Cor': '1 Corinthians', '1Cor': '1 Corinthians',
    '2 Cor': '2 Corinthians', '2Cor': '2 Corinthians',
    'Gal': 'Galatians', 'Eph': 'Ephesians', 'Phil': 'Philippians',
    'Col': 'Colossians',
    '1 Thess': '1 Thessalonians', '2 Thess': '2 Thessalonians',
    '1 Tim': '1 Timothy', '2 Tim': '2 Timothy', 'Tit': 'Titus',
    'Phlm': 'Philemon', 'Heb': 'Hebrews', 'Jas': 'James',
    '1 Pet': '1 Peter', '1Pet': '1 Peter',
    '2 Pet': '2 Peter', '2Pet': '2 Peter',
    '1 Jn': '1 John', '1Jn': '1 John',
    '2 Jn': '2 John', '3 Jn': '3 John',
    'Jude': 'Jude', 'Rev': 'Revelation', 'Apoc': 'Revelation',
}

SYSTEM_PROMPT = """\
You are an expert in medieval Latin philosophy and biblical theology, \
specializing in Anselm of Canterbury (1033–1109) and the Latin Vulgate Bible.

Your task: identify ALL biblical allusions in a given passage from Anselm's \
works, including:
1. direct_quote   — near-verbatim quotation from the Vulgate
2. paraphrase     — reworded version of a biblical passage
3. thematic_echo  — conceptual parallel or theological theme drawn from scripture

CRITICAL ORTHOGRAPHY NOTE: Anselm uses classical Latin where u/v are not \
distinguished; he writes 'u' for both consonantal and vocalic use:
  uirtus = virtus, uerbum = verbum, ueritas = veritas
  inuenire = invenire, uoluntas = voluntas, uia = via
Account for these when matching Anselm's vocabulary to Vulgate Latin.

CONTEXT: Anselm's Monologion and Proslogion deliberately avoid naming \
biblical sources (he argues from reason alone), but he was deeply saturated \
in scripture. Concepts, vocabulary, and argument structures often echo \
specific verses even without explicit citation.

Return ONLY a JSON object with exactly this structure:
{
  "allusions": [
    {
      "anselm_text": "the relevant phrase or sentence from Anselm",
      "book": "Full book name (e.g. Romans, Psalms, Genesis, 1 Corinthians)",
      "chapter": 1,
      "verse": 20,
      "type": "direct_quote|paraphrase|thematic_echo",
      "confidence": "high|medium|low",
      "explanation": "Brief explanation of the connection (1-2 sentences)"
    }
  ]
}

If no biblical allusions are found, return {"allusions": []}.\
"""


def make_user_prompt(section_text: str) -> str:
    max_chars = 3000
    if len(section_text) > max_chars:
        section_text = section_text[:max_chars] + ' ...[truncated]'
    return (
        "Identify all biblical allusions in this passage from "
        f"Anselm of Canterbury:\n\n{section_text}"
    )


def call_llm(
    section_text: str,
    api_key: str,
    model: str,
    cache: dict,
    cache_key: str,
) -> list:
    """
    Call LLM via OpenRouter API.
    Caches responses by cache_key to avoid redundant calls.
    Implements exponential backoff on HTTP 429.
    Returns list of raw allusion dicts from the LLM.
    """
    import urllib.error
    import urllib.request

    if cache_key in cache:
        return cache[cache_key]

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(section_text)},
        ],
        "temperature": 0.1,
    }).encode('utf-8')

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anselm-allusions",
        "X-Title": "Anselm Biblical Allusion Detector",
    }

    wait = 2
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                url, data=payload, headers=headers, method='POST'
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            # Guard: provider error returned as HTTP 200
            if 'choices' not in result or not result['choices']:
                err = result.get('error', result)
                print(f"    API error response: {err}", file=sys.stderr, flush=True)
                return []
            msg = result['choices'][0]['message']
            # Some reasoning models return content: null with output in reasoning_content
            content_str = (msg.get('content') or msg.get('reasoning_content') or '').strip()
            if not content_str:
                print(f"    Empty content in API response", file=sys.stderr, flush=True)
                return []
            # Strip markdown fences (some models ignore response_format: json_object)
            content_str = re.sub(r'^```(?:json)?\s*\n?', '', content_str, flags=re.IGNORECASE)
            content_str = re.sub(r'\n?```\s*$', '', content_str).strip()
            parsed = json.loads(content_str)
            allusions = parsed.get('allusions', [])
            cache[cache_key] = allusions
            return allusions
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(
                    f"    Rate limited (429), backing off {wait}s...",
                    file=sys.stderr, flush=True,
                )
                time.sleep(wait)
                wait = min(wait * 2, 60)
            else:
                body = e.read().decode('utf-8', errors='replace')
                print(
                    f"    HTTP {e.code} error: {e.reason} — {body[:200]}",
                    file=sys.stderr, flush=True,
                )
                return []
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}", file=sys.stderr, flush=True)
            print(f"    Raw content: {repr(content_str[:300])}", file=sys.stderr, flush=True)
            return []
        except Exception as e:
            print(f"    LLM error (attempt {attempt + 1}): {e}", file=sys.stderr)
            if attempt < 5:
                time.sleep(wait)
                wait = min(wait * 2, 60)

    return []


def call_llm_parallel(
    section_text: str,
    models: list,
    api_key: str,
    cache: dict,
    fname: str,
    sec_num: int,
) -> dict:
    """
    Call multiple worker models in parallel via ThreadPoolExecutor.
    Cache key per worker: "{fname}:{sec_num}:{model_slug}"
    Returns: {model: [allusions], ...}
    """
    def call_one(model: str):
        slug = model.replace('/', '_')
        cache_key = f"{fname}:{sec_num}:{slug}"
        return model, call_llm(section_text, api_key, model, cache, cache_key)

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {executor.submit(call_one, m): m for m in models}
        for future in concurrent.futures.as_completed(futures):
            model, allusions = future.result()
            results[model] = allusions

    return results



def union_worker_outputs(worker_outputs: dict) -> list:
    """
    Merge allusions from all workers without a judge.
    Deduplicates on (book, chapter, verse).
    Confidence is upgraded based on inter-model agreement:
      3+ models → high, 2 models → medium, 1 model → unchanged.
    Returns a list of raw allusion dicts suitable for process_llm_results().
    """
    seen: dict = {}  # (book, ch, vs) -> {'count': int, 'allusion': dict}
    for _model, allusions in worker_outputs.items():
        for a in allusions:
            book = normalize_book_name(a.get('book', ''))
            try:
                ch = int(a.get('chapter', 0))
                vs = int(a.get('verse', 0))
            except (ValueError, TypeError):
                continue
            if not book or ch <= 0 or vs <= 0:
                continue
            key = (book, ch, vs)
            if key in seen:
                seen[key]['count'] += 1
            else:
                a_copy = dict(a)
                a_copy['book'] = book
                seen[key] = {'count': 1, 'allusion': a_copy}

    result = []
    for _key, entry in seen.items():
        a = dict(entry['allusion'])
        count = entry['count']
        if count >= 3:
            a['confidence'] = 'high'
        elif count >= 2:
            a['confidence'] = 'medium'
        result.append(a)
    return result


def normalize_book_name(name: str) -> str:
    """Expand common abbreviations to full Vulgate book names."""
    if not name:
        return name
    name = name.strip()
    return BOOK_ALIASES.get(name, name)


def process_llm_results(
    allusions: list,
    section_text: str,
    verse_lookup: dict,
) -> list:
    """
    Post-process raw LLM allusions:
      - normalize book names
      - verify against verse_lookup
      - compute Jaccard similarity
    """
    results = []
    norm_section = normalize_medieval(section_text)

    for a in allusions:
        book = normalize_book_name(a.get('book', ''))
        try:
            chapter = int(a.get('chapter', 0))
            verse = int(a.get('verse', 0))
        except (ValueError, TypeError):
            continue

        if not book or chapter <= 0 or verse <= 0:
            continue

        key = (book, chapter, verse)
        vulgate_text = verse_lookup.get(key, '')
        verified = key in verse_lookup
        norm_vulgate = normalize_medieval(vulgate_text)
        sim = jaccard_similarity(norm_section, norm_vulgate) if vulgate_text else 0.0

        results.append({
            'book': book,
            'chapter': chapter,
            'verse': verse,
            'vulgate_text': vulgate_text,
            'ngram_score': 0,
            'similarity': sim,
            'method': 'llm',
            'allusion_type': a.get('type', 'thematic_echo'),
            'verified': verified,
            'llm_confidence': a.get('confidence', ''),
            'llm_explanation': a.get('explanation', ''),
            'anselm_text': a.get('anselm_text', ''),
            'judge_reasoning': a.get('judge_reasoning', ''),
        })

    return results


# ── Merge Results ──────────────────────────────────────────────────────────────

def merge_results(ngram_results: list, llm_results: list) -> list:
    """
    Deduplicate and merge n-gram and LLM results.
    Dedup key: (book, chapter, verse).
    When both pipelines find the same verse:
      - method becomes 'ngram+llm'
      - LLM explanation and type are adopted
      - Best similarity is kept
    """
    merged: dict = {}

    for r in ngram_results:
        key = (r['book'], r['chapter'], r['verse'])
        merged[key] = dict(r)

    for r in llm_results:
        key = (r['book'], r['chapter'], r['verse'])
        if key in merged:
            existing = merged[key]
            existing['method'] = 'ngram+llm'
            existing['llm_confidence'] = r.get('llm_confidence', '')
            existing['llm_explanation'] = r.get('llm_explanation', '')
            existing['anselm_text'] = r.get('anselm_text', '')
            existing['allusion_type'] = r.get('allusion_type', existing['allusion_type'])
            existing['judge_reasoning'] = r.get('judge_reasoning', '')
            if r.get('similarity', 0) > existing.get('similarity', 0):
                existing['similarity'] = r['similarity']
        else:
            merged[key] = dict(r)
            merged[key].setdefault('ngram_score', 0)
            merged[key].setdefault('judge_reasoning', '')

    return list(merged.values())


# ── Output ─────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    'file', 'section', 'section_title', 'anselm_text',
    'vulgate_book', 'chapter', 'verse', 'vulgate_text',
    'method', 'allusion_type', 'ngram_score', 'llm_confidence',
    'llm_explanation', 'similarity', 'verified', 'judge_reasoning',
]


def write_csv(rows: list, path: str) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}", file=sys.stderr)


def write_report(rows: list, path: str) -> None:
    lines = []
    sep = '=' * 70

    lines += [
        sep,
        "BIBLICAL ALLUSION DETECTION REPORT",
        "Anselm of Canterbury — Latin Works vs. Latin Vulgate",
        sep,
        "",
    ]

    verified = [r for r in rows if r.get('verified')]
    lines += [
        f"Total allusions detected : {len(rows)}",
        f"Verified against Vulgate : {len(verified)}",
        f"Unverified (LLM guesses) : {len(rows) - len(verified)}",
        "",
    ]

    # By detection method
    methods = Counter(r.get('method', '') for r in rows)
    lines.append("By detection method:")
    for method, count in methods.most_common():
        lines.append(f"  {method:<12}: {count}")
    lines.append("")

    # By file
    lines.append("By Anselm work:")
    by_file = Counter(r.get('file', '') for r in rows)
    for fname, count in by_file.most_common():
        lines.append(f"  {fname:<20}: {count}")
    lines.append("")

    # By allusion type
    lines.append("By allusion type:")
    by_type = Counter(r.get('allusion_type', '') for r in rows)
    for atype, count in by_type.most_common():
        lines.append(f"  {atype:<20}: {count}")
    lines.append("")

    # Bar chart: allusions per biblical book (top 20)
    lines.append("Allusions by biblical book (top 20):")
    by_book = Counter(r.get('vulgate_book', '') for r in rows)
    max_count = max(by_book.values()) if by_book else 1
    bar_width = 35
    for book, count in by_book.most_common(20):
        bar = '█' * int(round(count / max_count * bar_width))
        lines.append(f"  {book:<28} {bar:<35} {count}")
    lines.append("")

    # Top allusions by similarity
    lines.append("Top 15 allusions by similarity score:")
    lines.append('-' * 70)
    top = sorted(rows, key=lambda r: r.get('similarity', 0), reverse=True)[:15]
    for r in top:
        lines.append(
            f"  [{r.get('file', '')} §{r.get('section', '')}] "
            f"{r.get('vulgate_book', '')} {r.get('chapter', '')}:{r.get('verse', '')} "
            f"({r.get('method', '')}, sim={r.get('similarity', 0):.3f}, "
            f"{r.get('allusion_type', '')})"
        )
        if r.get('llm_explanation'):
            lines.append(f"    Explanation: {r['llm_explanation'][:100]}")
        if r.get('vulgate_text'):
            lines.append(f"    Vulgate: {r['vulgate_text'][:100]}")
        if r.get('anselm_text'):
            lines.append(f"    Anselm:  {r['anselm_text'][:100]}")
        lines.append("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Wrote report to {path}", file=sys.stderr)


# ── Main ───────────────────────────────────────────────────────────────────────

ANSELM_FILES = {
    'Monologion': 'data/Monologion.txt',
    'Proslogion': 'data/Proslogion.txt',
    'Pro_insipiente': 'data/Pro_insipiente.txt',
    'Responsio': 'data/Responsio.txt',
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Detect biblical allusions in Anselm of Canterbury',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # N-gram only (fast, no API cost):
  python detect_allusions.py --skip-llm

  # Single model (original behaviour):
  export OPENROUTER_API_KEY="sk-or-..."
  python detect_allusions.py --model deepseek/deepseek-chat-v3-0324

  # Two workers, union merge:
  python detect_allusions.py \\
    --models deepseek/deepseek-chat-v3-0324 openai/gpt-4o-mini

  # Specific files, LLM only:
  python detect_allusions.py --skip-ngram --files Proslogion Pro_insipiente
""",
    )
    parser.add_argument(
        '--api-key', metavar='KEY',
        help='OpenRouter API key (or set OPENROUTER_API_KEY env var)',
    )
    parser.add_argument(
        '--models', nargs='+', metavar='MODEL', default=None,
        help='Worker models to run in parallel (default: deepseek/deepseek-chat-v3-0324)',
    )
    parser.add_argument(
        '--model', metavar='MODEL', default=None,
        help='Deprecated alias for --models with a single value',
    )
    parser.add_argument(
        '--skip-llm', action='store_true',
        help='Skip LLM pipeline (n-gram only)',
    )
    parser.add_argument(
        '--skip-ngram', action='store_true',
        help='Skip n-gram pipeline (LLM only)',
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Ignore existing LLM cache (force fresh API calls)',
    )
    parser.add_argument(
        '--files', nargs='+', metavar='FILE',
        help='Process specific works only (e.g. Monologion Proslogion)',
    )
    parser.add_argument(
        '--vulgate', default='data/vulgate.json',
        help='Path to vulgate.json (default: data/vulgate.json)',
    )
    parser.add_argument(
        '--output-dir', default='results',
        help='Directory for output files (default: results/)',
    )
    parser.add_argument(
        '--cache-file', default='llm_cache.json',
        help='LLM cache filename (default: llm_cache.json)',
    )
    parser.add_argument(
        '--min-confidence', default='high', choices=['high', 'medium', 'low'],
        help='Minimum LLM confidence to retain (default: high). '
             'n-gram hits are always kept regardless.',
    )
    args = parser.parse_args()

    # Resolve worker models list
    if args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        models = ['openai/gpt-oss-120b:exacto']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve API key
    api_key = args.api_key or os.environ.get('OPENROUTER_API_KEY', '')
    if not args.skip_llm and not api_key:
        print(
            "Warning: No API key found. Set --api-key or OPENROUTER_API_KEY. "
            "Falling back to --skip-llm mode.",
            file=sys.stderr,
        )
        args.skip_llm = True

    # Load Vulgate
    print("Loading Vulgate...", file=sys.stderr)
    verse_lookup, norm_verses = load_vulgate(args.vulgate)

    # Build n-gram index
    ngram_index = None
    if not args.skip_ngram:
        print("Building n-gram index...", file=sys.stderr)
        ngram_index = build_ngram_index(norm_verses)

    # Load LLM cache
    cache_path = output_dir / args.cache_file
    cache: dict = {}
    if not args.no_cache and cache_path.exists():
        with open(cache_path, encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached LLM responses", file=sys.stderr)

    # Determine files to process
    files_to_process = dict(ANSELM_FILES)
    if args.files:
        files_to_process = {
            k: v for k, v in ANSELM_FILES.items()
            if k in args.files
        }
        unknown = set(args.files) - set(ANSELM_FILES)
        if unknown:
            print(f"Warning: unknown file names: {unknown}", file=sys.stderr)

    # Process each file
    all_rows: list = []

    for fname, fpath in files_to_process.items():
        if not Path(fpath).exists():
            print(f"Warning: {fpath} not found, skipping", file=sys.stderr)
            continue

        print(f"\n{'─' * 60}", file=sys.stderr)
        print(f"Processing {fname}...", file=sys.stderr)
        sections = parse_anselm_file(fpath)
        print(f"  Parsed {len(sections)} sections", file=sys.stderr)

        for section in sections:
            sec_num   = section['num']
            sec_title = section['title']
            sec_text  = section['text']

            if not sec_text.strip():
                continue

            n_results: list = []
            if not args.skip_ngram and ngram_index is not None:
                n_results = ngram_search(sec_text, ngram_index, verse_lookup)
                if n_results:
                    print(f"  §{sec_num}: {len(n_results)} n-gram hit(s)", file=sys.stderr)

            l_results: list = []
            if not args.skip_llm:
                if len(models) > 1:
                    print(
                        f"  §{sec_num}: querying {len(models)} model(s)...",
                        file=sys.stderr,
                    )
                    worker_outputs = call_llm_parallel(
                        sec_text, models, api_key, cache, fname, sec_num
                    )
                    raw_allusions = union_worker_outputs(worker_outputs)
                else:
                    slug = models[0].replace('/', '_')
                    cache_key = f"{fname}:{sec_num}:{slug}"
                    if cache_key in cache:
                        print(f"  §{sec_num}: using cached LLM response", file=sys.stderr)
                    else:
                        print(f"  §{sec_num}: querying LLM...", file=sys.stderr)
                    raw_allusions = call_llm(sec_text, api_key, models[0], cache, cache_key)

                # Persist cache after each section
                with open(cache_path, 'w', encoding='utf-8') as cf:
                    json.dump(cache, cf, ensure_ascii=False, indent=2)

                l_results = process_llm_results(raw_allusions, sec_text, verse_lookup)
                # Filter by confidence threshold; n-gram hits are kept regardless
                CONFIDENCE_RANK = {'high': 2, 'medium': 1, 'low': 0}
                min_rank = CONFIDENCE_RANK.get(args.min_confidence, 2)
                l_results = [
                    r for r in l_results
                    if CONFIDENCE_RANK.get(r.get('llm_confidence', 'low'), 0) >= min_rank
                ]
                if l_results:
                    print(f"  §{sec_num}: {len(l_results)} LLM allusion(s)", file=sys.stderr)

            merged = merge_results(n_results, l_results)
            for r in merged:
                all_rows.append({
                    'file': fname,
                    'section': sec_num,
                    'section_title': sec_title,
                    'anselm_text': r.get('anselm_text', ''),
                    'vulgate_book': r.get('book', ''),
                    'chapter': r.get('chapter', ''),
                    'verse': r.get('verse', ''),
                    'vulgate_text': r.get('vulgate_text', ''),
                    'method': r.get('method', ''),
                    'allusion_type': r.get('allusion_type', ''),
                    'ngram_score': r.get('ngram_score', 0),
                    'llm_confidence': r.get('llm_confidence', ''),
                    'llm_explanation': r.get('llm_explanation', ''),
                    'similarity': round(r.get('similarity', 0.0), 4),
                    'verified': r.get('verified', False),
                    'judge_reasoning': r.get('judge_reasoning', ''),
                })

    # Write outputs
    print(f"\n{'─' * 60}", file=sys.stderr)
    csv_path = output_dir / 'allusions.csv'
    report_path = output_dir / 'allusions_report.txt'
    write_csv(all_rows, str(csv_path))
    write_report(all_rows, str(report_path))
    print(f"\nDone. {len(all_rows)} total allusions detected.", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
