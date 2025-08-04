#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import os
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Import from the new 'google-genai' SDK
try:
    from google import genai
except ImportError:
    print("Error: The new Google GenAI SDK ('google-genai') is not installed.")
    print("Please install it using: pip install google-genai")
    sys.exit(1)

# --- SCRIPT CONSTANTS ---
SENTENCE_THRESHOLD = 1000
AVG_CLIPS_THRESHOLD = 5

# --- Configure Logging ---
logger = logging.getLogger(__name__)


# --- DATA LOADING ---


def read_tsv(file_path: Path) -> List[Dict[str, str]]:
    """
    Opens and reads a tab-separated values (TSV) file from the given path.
    """
    logger.info(f"Reading data from {file_path.name}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    except FileNotFoundError:
        logger.error(f"Fatal: The file {file_path} was not found.")
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"Fatal: An error occurred while reading {file_path}: {e}"
        )
        sys.exit(1)


# --- STATS CALCULATION ---


def get_hours(data: List[Dict[str, str]], durations: Dict[str, int]) -> float:
    """
    Calculates the total duration in hours for a specific subset of audio clips.
    """
    logger.info("Calculating clip hours for a subset of data...")
    total_ms = sum(durations.get(row["path"], 0) for row in data)
    return total_ms / (1000 * 60 * 60)


def get_demographics(data: List[Dict[str, str]]) -> Dict[str, Counter]:
    """
    Analyzes clip metadata to extract and count demographic information.
    """
    logger.info("Calculating demographic statistics (gender, age, accent)...")
    accents = [s for row in data if (s := row.get("accents", ""))]
    return {
        "gender": Counter(
            row.get("gender") for row in data if row.get("gender")
        ),
        "age": Counter(row.get("age") for row in data if row.get("age")),
        "accent": Counter(accents),
    }


def get_contributor_stats(data: List[Dict[str, str]]) -> Dict[str, int]:
    """
    Calculates the distribution of contributions per user.
    """
    logger.info("Calculating contributor statistics...")
    clips_per_contributor = Counter(row["client_id"] for row in data)
    bins = {"1-10": 0, "11-50": 0, "51-100": 0, "101-500": 0, ">500": 0}
    for count in clips_per_contributor.values():
        if 1 <= count <= 10:
            bins["1-10"] += 1
        elif 11 <= count <= 50:
            bins["11-50"] += 1
        elif 51 <= count <= 100:
            bins["51-100"] += 1
        elif 101 <= count <= 500:
            bins["101-500"] += 1
        else:
            bins[">500"] += 1
    return bins


def get_text_corpus_stats(
    validated_sentences: List[Dict[str, str]],
    validated_clips: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Performs a deep analysis of the dataset's text sentences.
    """
    logger.info("Analyzing text corpus statistics...")
    used_sentences = [
        row for row in validated_sentences if row.get("is_used") == "1"
    ]
    clips_counts = [int(row.get("clips_count", 0)) for row in used_sentences]
    all_text = [
        row["sentence"] for row in validated_clips if "sentence" in row
    ]

    total_tokens = sum(len(s.split()) for s in all_text)
    total_chars = sum(len(s) for s in all_text)

    return {
        "unique_sources": sorted(
            list(
                set(
                    row["source"]
                    for row in used_sentences
                    if row.get("source")
                )
            )
        ),
        "sentences_without_recording": sum(1 for c in clips_counts if c == 0),
        "average_clips_per_sentence": (
            round(sum(clips_counts) / len(clips_counts), 2)
            if clips_counts
            else 0
        ),
        "alphabet": sorted(list(set("".join(all_text)))),
        "sample_sentences": random.sample(all_text, k=min(5, len(all_text))),
        "average_sentence_length_tokens": (
            round(total_tokens / len(all_text), 1) if all_text else 0
        ),
        "average_sentence_length_chars": (
            round(total_chars / len(all_text), 1) if all_text else 0
        ),
    }


# --- PROMPT AND API CALL ---


def generate_prompt_for_llm(
    stats: Dict[str, Any], sentence_threshold: int, avg_clips_threshold: float
) -> str:
    """
    Constructs a highly specific, multi-part prompt to guide the Gemini LLM.
    """
    logger.info("Generating detailed prompt for the language model...")
    prompt_stats = json.loads(json.dumps(stats, default=lambda o: dict(o)))
    lang_name = stats["language"]["name"]

    return f"""
You are an expert AI assistant that creates datasheets for Mozilla Common Voice datasets.
Your task is to generate a comprehensive and accurate markdown file based on the statistical data provided below. Follow all instructions with extreme precision.

**STATISTICAL DATA:**
{json.dumps(prompt_stats, indent=2, ensure_ascii=False)}

**INSTRUCTIONS:**

**1. Main Title:**
Start with the title: `# Mozilla Common Voice: {lang_name} ({stats['language']['code']})`

**2. Language Section:**
Write a brief, encyclopedic introduction for the `{lang_name}` language. Include its language family, primary regions where it's spoken, and number of speakers.

**3. Clip & Sentence Statistics Section:**
Create a section titled `## Clip & Sentence Statistics`. In this section, state that the dataset contains a total of **{stats['clip_stats']['validated_hours']} validated hours** of speech from **{sum(stats['contributor_stats'].values())}** unique contributors. Then, create two markdown tables EXACTLY as follows, using the data provided:

First table (Clip Summary):
| Type                | Count | Hours  |
| ------------------- | ----: | -----: |
| Validated Clips     | {stats['clip_stats']['validated_count']:,} |   {stats['clip_stats']['validated_hours']:.2f} |
| Invalidated Clips   | {stats['clip_stats']['invalidated_count']:,} |   {stats['clip_stats']['invalidated_hours']:.2f} |
| **Total Clips**     | {stats['clip_stats']['total_count']:,} |   {stats['clip_stats']['total_hours']:.2f} |

Second table (Sentence Summary):
| Type                  |   Count |
| --------------------- | ------: |
| Validated Sentences   |   {stats['sentence_stats']['validated_count']:,} |
| Invalidated Sentences |     {stats['sentence_stats']['invalidated_count']:,} |
| **Total Sentences**   |     {stats['sentence_stats']['total_count']:,} |

**4. Demographic Information Section:**
Create a section `## Demographic Information`. Include the sentence: "Demographic information is self-reported by contributors and may not be representative of the entire speaker population."
- Create subsections for `### Age` and `### Gender` with their respective tables.
- Create a subsection `### Accent`. Inside this subsection, do the following:
  - First, create a table of the raw, user-reported accents and their counts from the `demographics.accent` data.
  - Second, immediately after that table, create a new table titled `#### English Translation of Accents`. This table should have two columns: 'Reported Accent' and 'Best-Effort English Translation/Explanation'. Provide a best-effort translation for each unique accent string. For city names (e.g., 'Almaty'), the translation is the name itself. For descriptive phrases (e.g., 'Оңтүстік'), provide the English meaning (e.g., 'Southern').

**5. Contributor Statistics Section:**
Create a section `## Contributor Statistics`. Include a table showing the distribution of clips recorded per contributor from `contributor_stats`.

**6. Text Corpus Section:**
Create a section `## Text Corpus`. Under this section, add the following bullet points using the data from `text_corpus`:
- **Total validated sentences:** {stats['sentence_stats']['validated_count']:,}
- **Sentences without a recording yet:** {stats['text_corpus']['sentences_without_recording']:,}
- **Average clips per validated sentence:** {stats['text_corpus']['average_clips_per_sentence']:.2f}
- **Average sentence length (tokens):** {stats['text_corpus']['average_sentence_length_tokens']:.1f}
- **Average sentence length (characters):** {stats['text_corpus']['average_sentence_length_chars']:.1f}

**7. Corpus Sources Subsection:**
Create a subsection `### Corpus Sources`. Analyze the `unique_sources` list. If a source is in a foreign language (e.g., 'мақал-мәтелдер'), identify the language and explain its meaning. If it's a URL or generic name ('sentence-collector'), explain it. Present this as a markdown table with 'Source' and 'Description' columns.

**8. Alphabet Subsection:**
Create a subsection `### Alphabet`. Use a markdown code block to display the characters from the `alphabet` list.

**9. Sample Sentences Subsection:**
Create a subsection `### Sample Sentences`. List the 5 sentences from `sample_sentences`.

**10. Community Links Section & Conditional Call to Action:**
Create a section `## Community Links`.
- Add a link to the Common Voice Page: `https://commonvoice.mozilla.org/{stats['language']['code']}`
- Add a link to the Sentence Collector: `https://commonvoice.mozilla.org/sentence-collector/#/{stats['language']['code']}/main`
- **Conditional Analysis:** If 'sentences_without_recording' is less than {sentence_threshold} OR 'average_clips_per_sentence' is greater than {avg_clips_threshold}, add a prominent note stating that this language is in **dire need of new sentences** to provide variety for voice contributors.

**11. Fun Fact Section:**
Create a new section `## Fun Fact`. In this section, provide one interesting, non-offensive, and culturally-sensitive fun fact about the `{lang_name}` language. The fact should be a short paragraph. You MUST cite the source for the fact (e.g., a URL to a reputable source like Wikipedia, a university website, or a linguistic database).

**12. Final Sections:**
- Create `## Datasheet Authors` and credit it with: 'This datasheet was generated automatically based on corpus statistics.'
- After everything else, add a final, encouraging call to action. Invite readers to help grow the dataset for `{lang_name}` by contributing their voice at the main site or by adding public domain sentences via the Sentence Collector tool.

**13. Formatting Rules (Strictly follow):**
- Maximum line width for all text is 79 characters. Use text wrapping for paragraphs.
- Use '$$$' as a delimiter for code blocks, NOT '```'.
- Right-align numbers in tables where appropriate.
"""


def call_gemini_api(prompt: str) -> str:
    """
    Initializes the Gemini client and sends a request to the API.
    """
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("Fatal: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    logger.info("Instantiating the GenAI Client...")
    try:
        client = genai.Client()

        logger.info(
            "Sending request to the Gemini API. This may take a moment..."
        )
        response = client.models.generate_content(
            model="models/gemini-2.5-pro", contents=prompt
        )

        logger.info("Successfully received response from the API.")
        return response.text
    except Exception as e:
        logger.error(
            f"Fatal: An error occurred during the Gemini API call: {e}"
        )
        sys.exit(1)


# --- MAIN EXECUTION ---


def main():
    """
    Main entry point and orchestrator for the datasheet generation script.

    USAGE:
        1. Make sure the 'google-genai' library is installed.
           (pip install -r requirements.txt)
        2. Set your Gemini API key as an environment variable:
           export GEMINI_API_KEY="YOUR_API_KEY_HERE"
        3. Run the script from the command line:
           python generate_datasheet.py --base_path /path/to/language_dir
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Generate a datasheet for a Mozilla Common Voice dataset using the Google GenAI SDK."
    )
    parser.add_argument(
        "--base_path",
        type=Path,
        required=True,
        help="Path to the language directory (e.g., ./cv-corpus-vX/kk)",
    )
    args = parser.parse_args()

    base_path = args.base_path
    if not base_path.is_dir():
        logger.error(
            f"Fatal: Provided path '{base_path}' is not a valid directory."
        )
        return

    lang_code = base_path.name
    lang_name_map = {
        "kk": "Kazakh",
        "en": "English",
        "nn-NO": "Norwegian Nynorsk",
        "tt": "Tatar",
    }
    lang_name = lang_name_map.get(lang_code, lang_code.upper())
    logger.info(
        f"Starting datasheet generation for language: {lang_name} ({lang_code})"
    )

    # --- Load all necessary data files ---
    validated_clips = read_tsv(base_path / "validated.tsv")
    invalidated_clips = read_tsv(base_path / "invalidated.tsv")
    all_clips_durations = read_tsv(base_path / "clip_durations.tsv")
    validated_sentences = read_tsv(base_path / "validated_sentences.tsv")
    invalidated_sentences_data = read_tsv(base_path / "invalidated.tsv")
    durations_map = {
        row["clip"]: int(row["duration[ms]"]) for row in all_clips_durations
    }

    # --- Aggregate all stats with the new structure ---
    validated_hours = round(get_hours(validated_clips, durations_map), 2)
    invalidated_hours = round(get_hours(invalidated_clips, durations_map), 2)

    stats = {
        "language": {"code": lang_code, "name": lang_name},
        "clip_stats": {
            "total_count": len(all_clips_durations),
            "validated_count": len(validated_clips),
            "invalidated_count": len(invalidated_clips),
            "validated_hours": validated_hours,
            "invalidated_hours": invalidated_hours,
            "total_hours": validated_hours + invalidated_hours,
        },
        "sentence_stats": {
            "validated_count": len(validated_sentences),
            "invalidated_count": len(invalidated_sentences_data),
            "total_count": len(validated_sentences)
            + len(invalidated_sentences_data),
        },
        "demographics": get_demographics(validated_clips),
        "contributor_stats": get_contributor_stats(validated_clips),
        "text_corpus": get_text_corpus_stats(
            validated_sentences, validated_clips
        ),
    }

    prompt = generate_prompt_for_llm(
        stats, SENTENCE_THRESHOLD, AVG_CLIPS_THRESHOLD
    )
    final_markdown = call_gemini_api(prompt)

    print("\n\n" + "=" * 30 + " RESULTS " + "=" * 30)
    print("\n--- PART 1: GATHERED STATISTICS (Data sent to LLM) ---\n")
    stats_for_printing = json.loads(
        json.dumps(stats, default=lambda o: dict(o))
    )
    print(json.dumps(stats_for_printing, indent=2, ensure_ascii=False))
    print("\n\n--- PART 2: FINAL MARKDOWN (Generated by Gemini API) ---\n")
    print(final_markdown)
    print("\n" + "=" * 28 + " END OF SCRIPT " + "=" * 28)


if __name__ == "__main__":
    main()
