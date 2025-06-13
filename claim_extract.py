import spacy
import re

def merge_segments(segments):
    """
    Merge segments where text starts lowercase into the previous segment.
    """
    merged = []
    for timerange, text in segments:
        if text and text[0].islower() and merged:
            # Continuation: extend previous segment
            prev_range, prev_text = merged[-1]
            start_prev, _ = prev_range.split('-', 1)
            end_curr = timerange.split('-', 1)[1]
            new_range = f"{start_prev}-{end_curr}"
            new_text = prev_text + " " + text
            merged[-1] = (new_range, new_text)
        else:
            merged.append((timerange, text))
    return merged

# Patterns for high-precision claim filtering
VERB_PATTERN = re.compile(r"\b(lose|gain|burn|get|makes?|is|will|can|do)\b", re.IGNORECASE)
INTRO_PATTERN = re.compile(r"^(one of|that'?s|my name|that's)", re.IGNORECASE)

def is_strong_claim(sent: str) -> bool:
    """
    Returns True if the sentence is a self-contained claim:
    - Does not start like an intro/ speaker ID
    - Contains an assertion verb
    """
    sent_l = sent.lower()
    # 1) filter out intros or speaker identifiers
    if INTRO_PATTERN.match(sent_l):
        return False
    # 2) must contain one of our assertion verbs
    return bool(VERB_PATTERN.search(sent_l))

def load_transcript(path="transcript.txt"):
    """Read time-stamped transcript into a list of (time_range, text)."""
    segments = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            timerange, text = line.split(":", 1)
            segments.append((timerange.strip(), text.strip()))
    return segments

def split_sentences(nlp, segments):
    """Use spaCy to split each segment into sentences."""
    sentences = []
    for timerange, text in segments:
        doc = nlp(text)
        for sent in doc.sents:
            sentences.append((timerange, sent.text))
    return sentences

def extract_entities(nlp, sentences):
    """
    Given a list of (timerange, sentence) pairs, returns a list of
    (timerange, sentence, entities), where entities is a list of
    (entity_text, entity_label) tuples from spaCy.
    """
    results = []
    for timerange, sent in sentences:
        doc = nlp(sent)
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        results.append((timerange, sent, ents))
    return results

def filter_claims(entity_results,
                           allowed_labels=None,
                           keyword_pattern=None):
    """
    Returns only those (timerange, sent, ents) where either:
    1) ents contains an entity with label in allowed_labels, OR
    2) sent matches keyword_pattern
    """
    if allowed_labels is None:
        allowed_labels = {"QUANTITY", "CARDINAL", "PERCENT"}

    # match any of these words, case-insensitive
    if keyword_pattern is None:
        keyword_pattern = re.compile(
            r"\b(calories?|pound(s)?|weight|energy)\b",
            flags=re.IGNORECASE
        )

    claims = []
    for timerange, sent, ents in entity_results:
        has_numeric = any(label in allowed_labels for _, label in ents)
        has_keyword = bool(keyword_pattern.search(sent))
        if has_numeric or has_keyword:
            claims.append((timerange, sent, ents))
    return claims

if __name__ == "__main__":
    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm")

    # Load and merge transcript segments
    segments = load_transcript("transcript.txt")
    segments = merge_segments(segments)
    sentences = split_sentences(nlp, segments)

    # 1) Extract all entities from each sentence
    entity_results = extract_entities(nlp, sentences)

    # 2) (Optional) Print out what we found for each sentence
    print("\n=== Entity tagging ===")
    for timerange, sent, ents in entity_results:
        print(f"{timerange} → {sent}")
        print("   Entities:", ents)
    print("\n=== Filtered candidate claims ===")

    # 3) Filter only those sentences with numeric/quantity entities
    claims = filter_claims(entity_results)

    # 4) Apply high-precision filter
    final_claims = [
        (t, s, e) for t, s, e in claims
        if is_strong_claim(s)
    ]

    # 5) Print filtered high-precision claims
    print("\n=== Final, high-precision claims ===")
    for timerange, sent, ents in final_claims:
        print(f"{timerange} → {sent}")