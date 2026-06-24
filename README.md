# Biblical Allusion Detector — Anselm of Canterbury

Detects direct quotes, paraphrases, and thematic echoes of the Latin Vulgate in Anselm's philosophical works (*Monologion*, *Proslogion*, *Pro insipiente*, *Responsio*).

## v2 — Validated analysis (current results)

The committed results are a **v2 re-analysis**. It re-runs detection through three complementary channels — contiguous word-sequence matching, harvesting of the quotation marks preserved in Anselm's own text, and a rarity-gated search for short distinctive phrases — and then validates every candidate by hand against both its context in Anselm and the surrounding verses in the Vulgate, keeping a match only where it carries genuine allusive sense.

Compared with the first-pass pipeline, v2:

- **removes coincidental matches** (generic Latin phrases that happened to align with an unrelated verse);
- **recovers genuine quotations the n-gram pass missed**, including several flagged only by Anselm's own quotation marks (e.g. 1 Cor 15:44, Matt 6:6, Job 3:24);
- **corrects Vulgate-vs-modern Psalm numbering** (the Fool is Vulgate Ps 13:1, not 14:1; the "nothing greater" formula resonates with Ps 144:3);
- **tiers every allusion by strength of evidence** — A (verbatim quotation), B (probable/paraphrase), C (distinctive short echo), D (thematic/conceptual);
- **reports scriptural density** (references per 1,000 words), separating explicit quotation from conceptual echo.

Two deliverables in `results/`:

- **[report.docx](results/report.docx)** — work-by-work written report: methodology, the density analysis, the argument's load-bearing texts, and side-by-side Anselm/Vulgate context for the key matches.
- **[data.xlsx](results/data.xlsx)** — the full row-level catalogue (textual tiers A–C and thematic tier D), a density sheet, and every reviewed-and-excluded candidate.

The sections below describe the underlying first-pass detector (`detect_allusions.py`), which still runs and regenerates the raw candidate set that v2 was built from.

## How it works

Two pipelines run in sequence and their results are merged:

1. **N-gram matching** — tokenises both texts with medieval Latin normalisation (`v→u`, `j→i`) and looks for shared word sequences. Fast, free, and precise for direct quotation.
2. **LLM via OpenRouter** — sends each section to a language model and asks it to identify paraphrases and thematic echoes that the n-gram pass would miss. Results are verified back against the Vulgate before being kept.

Running `--skip-llm` gives the n-gram baseline (66 hits across the four works). The first-pass LLM results were produced using `openai/gpt-oss-120b` as the LLM worker; the validated v2 catalogue above was built from that candidate set.

## Requirements

Python 3.9+ with no third-party dependencies. An [OpenRouter](https://openrouter.ai/) API key is needed for the LLM pipeline.

## Usage

```bash
# N-gram only — free, no API key needed
python detect_allusions.py --skip-llm

# Full run (n-gram + LLM)
export OPENROUTER_API_KEY="sk-or-..."
python detect_allusions.py

# Choose models
python detect_allusions.py --model deepseek/deepseek-chat-v3-0324
python detect_allusions.py --models deepseek/deepseek-chat-v3-0324 openai/gpt-4o-mini

# Specific works only
python detect_allusions.py --skip-llm --files Proslogion Pro_insipiente

# All options
python detect_allusions.py --help
```

Key flags:

| Flag | Description |
|---|---|
| `--skip-llm` | N-gram pipeline only |
| `--skip-ngram` | LLM pipeline only |
| `--min-confidence high\|medium\|low` | Filter LLM results by confidence (default: `high`) |
| `--no-cache` | Ignore cached LLM responses and re-query |
| `--output-dir DIR` | Output directory (default: `results/`) |
| `--vulgate PATH` | Path to Vulgate JSON (default: `data/vulgate.json`) |

## Output

The **committed, authoritative results** are the v2 validated analysis: `results/report.docx` and `results/data.xlsx` (see the v2 section above).

Running `detect_allusions.py` regenerates the first-pass candidate set into the output directory:

| File | Contents |
|---|---|
| `allusions.csv` | One row per candidate: Anselm work, section, Vulgate reference, matched text, method, confidence |
| `allusions_report.txt` | Summary statistics: counts by work, biblical book, allusion type |
| `llm_cache.json` | Cached LLM responses (re-query with `--no-cache`) |

These raw outputs are regenerable and no longer committed; the validated `data.xlsx` supersedes them.

## Project layout

```
detect_allusions.py        First-pass detector (n-gram + LLM)
data/
  vulgate.json          Latin Vulgate (73 books, 35,811 verses)
  Monologion.txt
  Proslogion.txt
  Pro_insipiente.txt
  Responsio.txt
results/
  report.docx           v2 validated analysis — written report
  data.xlsx             v2 validated analysis — full dataset
```

## Data sources

- Vulgate: [emilekm2142/vulgate-bible-full-text](https://github.com/emilekm2142/vulgate-bible-full-text/blob/master/bible.json)
- Anselm Latin texts: [homepages.uc.edu/~martinj/Latin/Anselm/](https://homepages.uc.edu/~martinj/Latin/Anselm/)
