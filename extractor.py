"""
extractor.py — Entity & Fact Extraction
════════════════════════════════════════

CONCEPT:
  The extractor is the intelligence layer that turns raw conversation text
  into structured knowledge for the semantic memory tier.

  Given: "Hi, I'm Akshat and I'm based in Hyderabad. I prefer Python."
  It extracts:
    - entity: "user", attribute: "name",     value: "Akshat"
    - entity: "user", attribute: "location", value: "Hyderabad"
    - entity: "user", attribute: "lang_pref",value: "Python"

  Two extraction strategies:
  1. spaCy NER (Named Entity Recognition) — finds PERSON, GPE, ORG, etc.
  2. Regex/pattern rules — catches "my name is X", "I prefer X", etc.
     (spaCy misses many personal preference statements that don't look like
      named entities, so we combine both approaches)

HOW spaCy NER WORKS:
  spaCy's en_core_web_sm model is a trained neural pipeline:
    text → tokenizer → tagger → dependency parser → NER

  NER labels we care about:
  - PERSON  → names ("Akshat", "Elon Musk")
  - GPE     → geopolitical entities → locations ("Hyderabad", "India")
  - ORG     → organizations ("Microsoft", "Anthropic")
  - PRODUCT → product names ("Azure", "ChromaDB")
  - WORK_OF_ART → project names (sometimes)
  - DATE    → dates/times ("last Tuesday", "2024")

  spaCy gives us: entity text + label + confidence (via token scores)

INSTALL NOTE:
  After pip install spacy, you must download the model:
  python -m spacy download en_core_web_sm
"""

import re
import logging
from typing import Optional
import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)


# ── Pattern Rules for Personal Facts ────────────────────────────────────────
# spaCy NER is great for detecting WHAT is an entity, but it misses
# relationships like "I prefer X" or "I live in Y".
# These regex patterns capture those relationship statements directly.

# Each rule: (pattern, entity, attribute, value_group_index)
# The pattern captures the VALUE in a named group "value"
PERSONAL_FACT_PATTERNS = [
    # ── IDENTITY / NAME ───────────────────────────────────────────────────────
    # "my name is X" — the safest anchor, no ambiguity
    (r"my name is\s+(?P<value>[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]+)?)",
     "user", "name"),

    # "I'm X" / "I am X" / "call me X" — only if followed by a proper noun.
    # Negative lookahead blocks filler words that caused false positives:
    #   "I'm not sure"  → blocked by "not"
    #   "I'm happy"     → blocked by "happy"
    #   "I am an AI"    → blocked by "an"
    #   "I'm going to"  → blocked by "going"
    (r"(?:i'm|i am|call me)\s+(?!(?:not|an|a|the|here|happy|sorry|going|thinking|afraid|just|also|still|really|very|quite|so|now|trying|looking|working|building|using|based|from|in|at|with|for|able|ready|sure|glad|excited|pleased|honored)\b)(?P<value>[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20})?)",
     "user", "name"),

    # ── LOCATION ──────────────────────────────────────────────────────────────
    # "I'm based in X" / "I live in X" / "I'm from X"
    # Well-anchored — the "based/located/living/from in" prefix prevents false matches
    (r"(?:i(?:'m| am) (?:based|located|living|from) in|i live in|i(?:'m| am) from)\s+(?P<value>[A-Z][a-zA-Z\s]{2,30}?)(?:\.|,|\s+and\s|\s+with\s|$)",
     "user", "location"),

    # ── ROLE ──────────────────────────────────────────────────────────────────
    # Matches "I'm a/an X" and "I work as a/an X" where X ends with a known job title.
    # Key fix: {0,40}? (not {2,40}?) so bare "developer" (no prefix words) is captured.
    # "I'm not a developer" blocked by (?!not\s) negative lookahead.
    (r"(?:i(?:'m| am) a(?:n)?\s+|i work as a(?:n)?\s+)(?!not\s)(?P<value>[a-zA-Z\s]{0,40}?(?:engineer|developer|manager|designer|scientist|analyst|architect|researcher|consultant|lead|intern))\b",
     "user", "role"),

    # ── EXPERIENCE ────────────────────────────────────────────────────────────
    # Number-anchored — very safe, no false positives possible
    (r"(?P<value>\d+)\+?\s+years?\s+(?:of\s+)?experience",
     "user", "experience_years"),

    # ── TECH PREFERENCE ───────────────────────────────────────────────────────
    # Only explicit preference verbs: prefer/love/enjoy/mainly use/primarily use.
    # Removed "i use" and "i like" — too broad.
    # "I use X for Y" goes to tool_usage instead.
    # Tech preference: value must be a plausible technology name, not an adjective/adverb.
    # Negative lookahead on the VALUE blocks common English words:
    #   "I prefer very" → "very" is in blocklist → no match ✅
    #   "I prefer concise" → "concise" is in blocklist → no match ✅
    #   "I prefer Python" → "Python" not in blocklist → matches ✅
    #   "I prefer not to" → "not" is in blocklist → no match ✅
    (r"(?:i (?:prefer|love|enjoy|mainly use|primarily use)|my (?:preferred|favourite|favorite|go-to) (?:language|framework|tool|stack) is)"
     r"\s+(?!(?:very|quite|really|extremely|highly|fairly|rather|pretty|somewhat|concise|simple|clean|clear|fast|slow|easy|hard|good|bad|better|best|not|no|never|always|often|more|most|less|least|much|many|few|any|all|both|each|every|some|other|another|such|same|different|various|new|old|big|small|large|short|long|high|low|first|last|next|this|that|these|those|it|its|my|your|his|her|our|their)\b)"
     r"(?P<value>[A-Z][A-Za-z0-9#+.\-]{1,20}|\b(?:python|javascript|typescript|golang|rust|java|ruby|swift|kotlin|scala|bash|sql|go|c\+\+|c#)\b)",
     "user", "tech_preference"),

    # ── PROJECT NAME ──────────────────────────────────────────────────────────
    # "I'm building X" or "my project is/called/named X".
    # Removed bare "working on" — matches too many assistant phrases.
    # Value must start with uppercase (proper project name).
    # Optional article (a/an/the) before the name is swallowed.
    # PROJECT NAME — compiled WITHOUT re.IGNORECASE (unlike other patterns).
    # Why: Python's re.IGNORECASE makes [A-Z] match lowercase too, so
    # "I'm building understanding" would match "understanding" as a project name.
    # By dropping IGNORECASE and writing [Ii]/[Mm]y explicitly, [A-Z] in the
    # VALUE group strictly enforces that project names start with a capital letter.
    # "I'm building a project called X" and "my project is/called/named X" all match.
    # "I'm building something new" / "my project is going well" do NOT match.
    # NOTE: this tuple has a 4th element (False) = do NOT compile with IGNORECASE.
    (r"(?:[Ii](?:'m| am| AM) building(?: a project called)?|[Ii](?:'m| am| AM) working on(?: a project called)?|[Mm]y project (?:is called|is named|is|called))\s+(?:a |an |the )?(?P<value>[A-Z][A-Za-z0-9\-_]{1,39}(?:\s+[A-Z][A-Za-z0-9\-_]{1,40})?)",
     "project", "name", False),

    # ── TOOL USAGE ────────────────────────────────────────────────────────────
    # "I use X for Y" — explicit "I use" only (removed "we use" and "using").
    # Negative lookahead blocks pronouns/articles: "I use this/that/my/the/it/a for"
    # Value allows one space to catch "VS Code", "IntelliJ IDEA" etc.
    (r"\bi use\s+(?!this\b|that\b|my\b|the\b|it\b|a\b|an\b|these\b|those\b)(?P<value>[A-Z][A-Za-z0-9\-_\.]+(?:\s+[A-Za-z0-9\-_\.]+)?)\s+(?:for|as)\b",
     "user", "tool_usage"),
]

# Compile patterns once at module load (performance).
# Each tuple is (pattern, entity, attribute) or (pattern, entity, attribute, use_ignorecase).
# use_ignorecase defaults to True; set False for patterns where the VALUE group
# must strictly enforce uppercase (e.g. project names).
COMPILED_PATTERNS = [
    (
        re.compile(pattern, re.IGNORECASE if (len(t) < 4 or t[3]) else 0),
        t[1],
        t[2],
    )
    for t in PERSONAL_FACT_PATTERNS
    for pattern in [t[0]]
]


# ── Extractor Class ──────────────────────────────────────────────────────────

class Extractor:
    """
    Extracts entities and facts from conversation text using spaCy + rules.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Load the spaCy model.

        Args:
            model: spaCy model name. en_core_web_sm is ~12MB and fast.
                   For better accuracy (at the cost of speed), use en_core_web_lg.
        """
        logger.info(f"Loading spaCy model: {model}")
        try:
            self._nlp: Language = spacy.load(model)
            logger.info("spaCy model loaded successfully.")
        except OSError:
            logger.error(
                f"spaCy model '{model}' not found. "
                f"Run: python -m spacy download {model}"
            )
            raise

    # ── Entity Extraction ────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> list[dict]:
        """
        Run spaCy NER on text and return named entities.

        Returns:
            List of entity dicts: {text, label, start, end}

        Example:
            entities = extractor.extract_entities("I'm Akshat from Hyderabad")
            # → [
            #     {"text": "Akshat", "label": "PERSON", "start": 4, "end": 10},
            #     {"text": "Hyderabad", "label": "GPE", "start": 16, "end": 25},
            #   ]
        """
        doc = self._nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text.strip(),
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

        return entities

    def get_entity_strings(self, text: str) -> list[str]:
        """
        Return just the entity text strings (for working memory tagging).

        Example: ["Akshat", "Hyderabad", "Python"]
        """
        entities = self.extract_entities(text)
        return [e["text"] for e in entities]

    # ── Fact Extraction ──────────────────────────────────────────────────────

    def extract_facts(self, text: str, role: str = "user") -> list[dict]:
        """
        Extract structured facts from a conversation turn.

        Combines two strategies:
        1. spaCy NER → maps entity labels to fact attributes
        2. Regex patterns → catches personal relationship statements

        Args:
            text: the conversation text to analyze
            role: "user" or "assistant" — we typically only extract facts
                  from user turns (the assistant doesn't tell us about itself)

        Returns:
            List of fact dicts:
            {entity, attribute, value, confidence, source}

        Example:
            facts = extractor.extract_facts(
                "I'm Akshat, a software engineer based in Hyderabad. "
                "I'm building MemoryOS using Python and ChromaDB."
            )
            # → [
            #   {entity:"user", attribute:"name", value:"Akshat", confidence:0.9, source:"spacy_PERSON"},
            #   {entity:"user", attribute:"location", value:"Hyderabad", confidence:0.85, source:"spacy_GPE"},
            #   {entity:"user", attribute:"role", value:"software engineer", confidence:0.8, source:"pattern"},
            #   {entity:"project", attribute:"name", value:"MemoryOS", confidence:0.75, source:"pattern"},
            # ]
        """
        # We only extract facts from user turns, not assistant responses.
        # The assistant's output is a response, not a self-disclosure.
        if role != "user":
            return []

        facts = []

        # ── Strategy 1: spaCy NER ────────────────────────────────────────
        spacy_facts = self._extract_from_spacy(text)
        facts.extend(spacy_facts)

        # ── Strategy 2: Regex Pattern Rules ──────────────────────────────
        pattern_facts = self._extract_from_patterns(text)
        facts.extend(pattern_facts)

        # Deduplicate: if both strategies found the same entity+attribute,
        # keep the higher-confidence one
        facts = self._deduplicate_facts(facts)

        logger.debug(f"Extracted {len(facts)} facts from text ({len(text)} chars)")
        return facts

    def _extract_from_spacy(self, text: str) -> list[dict]:
        """
        Map spaCy NER labels to semantic memory fact attributes.

        spaCy label → semantic memory mapping:
        - PERSON  → user.name (if first person) or person.name
        - GPE     → user.location (assumed unless context says otherwise)
        - ORG     → organization.name
        - PRODUCT → technology.name
        """
        doc = self._nlp(text)
        facts = []

        # Check if text is first-person (user talking about themselves).
        # STRICT check: require sentence-opening first-person markers, not just "my "
        # which appears in assistant phrases like "Let me explain my reasoning."
        text_lower = text.lower()
        is_first_person = any(
            marker in text_lower
            for marker in [
                "i'm ", "i am ", "i've ", "i have ", "i use ", "i work ",
                "i live ", "i prefer ", "i like ", "my name is",
                "call me ", "i was ", "i do ", "i build ", "i built ",
            ]
        )

        for ent in doc.ents:
            entity_text = ent.text.strip()

            # Skip very short entities (likely noise) and numbers alone
            if len(entity_text) < 2 or entity_text.isdigit():
                continue

            if ent.label_ == "PERSON" and is_first_person:
                # "I'm Akshat" → user.name
                # Block known tech terms that spaCy sometimes tags as PERSON.
                # e.g. "I prefer Python" → spaCy tags "Python" as PERSON → wrong.
                _TECH_TERMS = {
                    "python", "javascript", "typescript", "java", "golang", "rust",
                    "chromadb", "sqlite", "postgresql", "mongodb", "redis", "docker",
                    "kubernetes", "azure", "aws", "gcp", "linux", "flask", "django",
                    "fastapi", "react", "angular", "vue", "ollama", "llama", "mistral",
                    "qwen", "git", "github", "windows", "macos", "ubuntu", "debian",
                    "tensorflow", "pytorch", "numpy", "pandas", "spark", "kafka",
                }
                if entity_text.lower() in _TECH_TERMS:
                    continue  # skip — not a person name
                # Also skip single-word entities that are ALL CAPS (acronyms, not names)
                if entity_text.isupper() and len(entity_text) <= 4:
                    continue
                facts.append({
                    "entity": "user",
                    "attribute": "name",
                    "value": entity_text,
                    "confidence": 0.85,
                    "source": "spacy_PERSON",
                })

            elif ent.label_ == "GPE" and is_first_person:
                # "I'm based in Hyderabad" → user.location
                facts.append({
                    "entity": "user",
                    "attribute": "location",
                    "value": entity_text,
                    "confidence": 0.80,
                    "source": "spacy_GPE",
                })

            elif ent.label_ == "ORG" and is_first_person:
                # "I work at Microsoft" → organization.name
                # Guard 1: require is_first_person — we only care about orgs the
                #          USER mentions, not every org in the conversation.
                # Guard 2: skip timezone/calendar acronyms and very short noise.
                _NOISE_ORGS = {
                    "ist", "gmt", "utc", "pst", "est", "cst", "mst", "bst",
                    "am", "pm", "ai", "ml", "api", "sdk", "ide", "os", "ui",
                }
                if entity_text.lower() in _NOISE_ORGS:
                    continue
                # Skip pure-acronym entities under 4 chars (too noisy)
                if entity_text.isupper() and len(entity_text) < 4:
                    continue
                facts.append({
                    "entity": "organization",
                    "attribute": "name",
                    "value": entity_text,
                    "confidence": 0.75,
                    "source": "spacy_ORG",
                })

            elif ent.label_ == "PRODUCT":
                # "using ChromaDB" → technology.name
                facts.append({
                    "entity": "technology",
                    "attribute": "name",
                    "value": entity_text,
                    "confidence": 0.70,
                    "source": "spacy_PRODUCT",
                })

        return facts

    def _extract_from_patterns(self, text: str) -> list[dict]:
        """
        Apply regex pattern rules to catch personal relationship statements
        that spaCy NER misses.

        Example: "I prefer Python" → spaCy sees "Python" as PRODUCT,
        but doesn't know it's a USER PREFERENCE. Our pattern catches this.
        """
        facts = []

        for pattern, entity, attribute in COMPILED_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    value = match.group("value").strip()
                    # Clean up trailing punctuation
                    value = value.rstrip(".,!?;:")

                    if len(value) < 2:
                        continue

                    facts.append({
                        "entity": entity,
                        "attribute": attribute,
                        "value": value,
                        "confidence": 0.75,  # patterns are less reliable than NER
                        "source": "pattern",
                    })
                    logger.debug(f"Pattern match: {entity}.{attribute} = {value!r}")

                except IndexError:
                    # Regex group didn't capture — skip
                    pass

        return facts

    def _deduplicate_facts(self, facts: list[dict]) -> list[dict]:
        """
        Remove duplicate entity+attribute+value tuples, keeping highest confidence.

        When both spaCy and a pattern find the same fact (e.g., user.name = Akshat),
        we keep the higher-confidence version.
        
        CRITICAL: Include value in the deduplication key so that different values
        are NOT discarded. This allows storing multiple technologies in tech stacks.
        
        Example:
        - Same fact, different sources → keep highest confidence
          {entity:"technology", attribute:"name", value:"Python", confidence:0.70}
          {entity:"technology", attribute:"name", value:"Python", confidence:0.85}
          → Keep the 0.85 version
        
        - Different values → keep BOTH (not deduplicated)
          {entity:"technology", attribute:"name", value:"Python", confidence:0.70}
          {entity:"technology", attribute:"name", value:"ChromaDB", confidence:0.70}
          → Keep both (they have different values)
        """
        seen: dict[tuple, dict] = {}  # (entity, attribute, value) → best fact

        for fact in facts:
            key = (fact["entity"], fact["attribute"], fact["value"])
            if key not in seen or fact["confidence"] > seen[key]["confidence"]:
                seen[key] = fact

        return list(seen.values())

    # ── Memory Classification ────────────────────────────────────────────────

    def classify_memory_tier(self, content: str) -> str:
        """
        Decide which memory tier is best for a given piece of content.

        Used by the remember() tool to auto-route storage.

        Rules:
        - If content looks like a structured fact (short, declarative) → "semantic"
        - If content is a conversation excerpt or narrative → "episodic"
        - Very short scratchpad items → "working" (but usually tools don't call this)

        Returns: "working" | "episodic" | "semantic"
        """
        content_lower = content.lower().strip()
        word_count = len(content.split())

        # Short structured statements with "is" or "=" → semantic
        if word_count <= 10 and any(
            marker in content_lower
            for marker in ["is ", "= ", "lives in", "works at", "prefers", "name:"]
        ):
            return "semantic"

        # Long narrative content → episodic
        if word_count > 20:
            return "episodic"

        # Medium length with personal pronouns → episodic (personal narrative)
        if any(pronoun in content_lower for pronoun in ["said", "mentioned", "told", "asked"]):
            return "episodic"

        # Default: episodic (safest choice — vector search is flexible)
        return "episodic"

    def compute_importance(self, content: str, entities: list[dict]) -> float:
        """
        Compute an importance score (0.0 to 1.0) for a piece of content.

        Higher importance = prioritized in recall when scores are equal.

        Heuristic rules:
        - Technical problems (bugs/errors) → HIGH importance (0.7+)
        - Personal identity facts → high importance (0.7+)
        - Goals and project work → medium-high importance (0.6+)
        - Named entities → boost (0.05 per entity)
        - Generic greetings → low importance penalty
        - Short content → low importance penalty

        This ensures critical development issues are stored even if brief,
        while enabling cross-session problem tracking.
        """
        score = 0.5  # baseline

        content_lower = content.lower()

        # CRITICAL: Technical problems → very high importance
        # Issues like "I hit a bug" or "Error: connection failed" MUST be remembered
        # Boost is 0.35 so even a one-liner bug gets 0.85 importance
        if any(word in content_lower for word in [
            "error", "bug", "crash", "fail", "broke", "broken", "exception",
            "threw", "throwing", "traceback", "stack overflow", "segfault",
            "issue", "problem", "fix", "fixed", "resolved", "solution",
            "debug", "debugging"
        ]):
            score += 0.35

        # Boost: has named entities (the content is about something specific)
        entity_boost = min(len(entities) * 0.05, 0.2)
        score += entity_boost

        # Boost: personal identity facts
        if any(word in content_lower for word in ["my name", "i am", "i'm", "i live"]):
            score += 0.2

        # Boost: goals and preferences (useful to remember long-term)
        if any(word in content_lower for word in ["want to", "prefer", "goal", "building", "working on", "learning"]):
            score += 0.15

        # Boost: deliverables and accomplishments
        if any(word in content_lower for word in ["completed", "finished", "shipped", "deployed", "released", "launched"]):
            score += 0.15

        # IMPORTANT: Don't heavily penalize short technical content
        # A one-line bug report "ChromaDB threw error X" is more important than a long greeting
        # Only penalize if it's BOTH short AND has no technical keywords
        if len(content.split()) < 5 and not any(word in content_lower for word in [
            "error", "bug", "crash", "fail", "broke", "broken", "exception"
        ]):
            score -= 0.1

        # Penalty: generic conversational fluff
        if any(word in content_lower for word in ["hello", "hi", "thanks", "okay", "sure", "yes", "no"]):
            score -= 0.15

        return max(0.0, min(1.0, score))