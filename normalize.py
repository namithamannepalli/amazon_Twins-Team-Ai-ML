"""
Text normalization helpers.

Why this matters: "Acme Robotics Inc." and "Acme Robotics Incorporated" are the
same business, but a computer sees them as completely different strings unless
we clean them up first. Every function here takes messy text and makes it
more comparable.
"""

import re

from unidecode import unidecode

# Common legal-entity suffixes that don't help identify a business and just
# add noise (feel free to add more you notice in your data).
LEGAL_SUFFIXES = [
    "incorporated", "inc", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "group", "holdings", "llp", "plc", "pvt", "private",
]

# Common street-address abbreviations. Expanding them to a single canonical
# form means "St" and "Street" stop looking different to the model.
STREET_ABBREVIATIONS = {
    "st": "street",
    "ave": "avenue",
    "rd": "road",
    "dr": "drive",
    "blvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
}


def basic_clean(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. The foundation
    every other normalization step builds on."""
    if text is None:
        return ""
    # Transliterate non-Latin scripts (Devanagari, Tamil, ...) to Latin so that
    # native-script names can be compared with English ones.
    text = unidecode(str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # collapse multiple spaces
    return text


def normalize_name(name: str) -> str:
    """Clean a business name and strip legal suffixes like 'inc' / 'llc'."""
    text = basic_clean(name)
    tokens = [t for t in text.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    """Clean an address and expand common street abbreviations."""
    text = basic_clean(address)
    tokens = [STREET_ABBREVIATIONS.get(t, t) for t in text.split()]
    return " ".join(tokens)


def name_tokens(name: str) -> set:
    """Word-set of a normalized name, used for token-overlap features."""
    return set(normalize_name(name).split())


def address_tokens(address: str) -> set:
    """Word-set of a normalized address, used for token-overlap features."""
    return set(normalize_address(address).split())


def normalize_country(country: str) -> str:
    """Normalize a country label for exact-match blocking. Deliberately does
    NOT hardcode any specific country names — it works the same whether the
    value is 'US', 'India', 'France', or anything else, so it stays correct
    even when the test set introduces a country never seen in training."""
    return basic_clean(country)
