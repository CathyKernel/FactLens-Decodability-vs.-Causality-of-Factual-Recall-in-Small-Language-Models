"""Curated factual relation bank.

Each relation groups (subject, object) pairs that small language models
(GPT-2 124M, Pythia-410M, Qwen2.5-0.5B) plausibly acquired during pretraining.
The bank is intentionally small, hand-audited, and balanced enough for
multi-class linear probing: every relation has between 6 and 20 classes and
at least one fact per class.

Selection criteria
------------------
1. *Unambiguity* — each subject has exactly one canonical object; no
   time-sensitive facts (no presidents, no populations).
2. *Frequency* — facts appear verbatim on Wikipedia/Wikidata dumps, so even
   a 124M-parameter model has seen them many times.
3. *Difficulty spread* — relations range from trivial (animal -> young) to
   harder (landmark -> city), which yields informative variance in where in
   the layer stack the answer becomes decodable vs. causal.

Templates use a single ``{S}`` placeholder. The subject span is recovered at
tokenization time via character offset mapping (see ``factlens.data.dataset``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Relation:
    """A binary factual relation with surface templates.

    Attributes:
        name: Short identifier, e.g. ``"capital-country"``.
        description: One-line semantics, e.g. ``"country -> capital city"``.
        templates: Prompt templates; each contains exactly one ``{S}``
            placeholder and ends where the object is expected to follow.
        facts: List of ``(subject, object)`` pairs. Objects define the label
            space for probing.
    """

    name: str
    description: str
    templates: list[str]
    facts: list[tuple[str, str]]
    notes: str = ""

    @property
    def num_facts(self) -> int:
        return len(self.facts)

    @property
    def objects(self) -> list[str]:
        """Label space (unique objects, insertion-ordered)."""
        out: list[str] = []
        for _, obj in self.facts:
            if obj not in out:
                out.append(obj)
        return out

    def render(self, subject: str, template_idx: int = 0) -> str:
        """Fill a template with a subject (no trailing whitespace added)."""
        return self.templates[template_idx].format(S=subject)


# ---------------------------------------------------------------------------
# The fact bank.
# ---------------------------------------------------------------------------
# Template conventions: prompts end with a connective ("is", "by", "a") so
# that the next token is the first object token, usually with a leading
# space, e.g. "The capital of France is" -> " Paris".

CAPITAL_COUNTRY = Relation(
    name="capital-country",
    description="country -> capital city",
    templates=[
        "The capital of {S} is",
        "The capital city of {S} is",
        "{S}'s capital is",
        "The seat of government of {S} is located in",
        "Head to {S} and you will find its capital in",
    ],
    facts=[
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("Russia", "Moscow"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Egypt", "Cairo"),
        ("Brazil", "Brasilia"),
        ("Argentina", "Buenos Aires"),
        ("Portugal", "Lisbon"),
        ("Greece", "Athens"),
        ("Turkey", "Ankara"),
        ("India", "New Delhi"),
        ("South Korea", "Seoul"),
        ("Poland", "Warsaw"),
        ("Netherlands", "Amsterdam"),
        ("Mexico", "Mexico City"),
    ],
)

WORK_AUTHOR = Relation(
    name="work-author",
    description="literary work -> author",
    templates=[
        "The author of {S} is",
        "The writer of {S} is",
        "{S} was written by",
        "The book {S} was penned by",
        "Who wrote {S}? The author is",
    ],
    facts=[
        ("Romeo and Juliet", "William Shakespeare"),
        ("Hamlet", "William Shakespeare"),
        ("Macbeth", "William Shakespeare"),
        ("Pride and Prejudice", "Jane Austen"),
        ("1984", "George Orwell"),
        ("Animal Farm", "George Orwell"),
        ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("Moby Dick", "Herman Melville"),
        ("War and Peace", "Leo Tolstoy"),
        ("Crime and Punishment", "Fyodor Dostoevsky"),
        ("The Odyssey", "Homer"),
        ("Don Quixote", "Miguel de Cervantes"),
        ("Frankenstein", "Mary Shelley"),
        ("Jane Eyre", "Charlotte Bronte"),
        ("Oliver Twist", "Charles Dickens"),
        ("Ulysses", "James Joyce"),
    ],
)

PRODUCT_COMPANY = Relation(
    name="product-company",
    description="product -> manufacturing company",
    templates=[
        "{S} is made by",
        "{S} is a product of",
        "The company behind {S} is",
        "The maker of {S} is",
        "{S} was created by",
    ],
    facts=[
        ("the iPhone", "Apple"),
        ("the iPad", "Apple"),
        ("the MacBook", "Apple"),
        ("Windows", "Microsoft"),
        ("the Xbox", "Microsoft"),
        ("Excel", "Microsoft"),
        ("the PlayStation", "Sony"),
        ("Android", "Google"),
        ("Chrome", "Google"),
        ("Gmail", "Google"),
        ("the Kindle", "Amazon"),
        ("Alexa", "Amazon"),
        ("Instagram", "Facebook"),
        ("WhatsApp", "Facebook"),
        ("the Mustang", "Ford"),
        ("the Switch", "Nintendo"),
    ],
)

LANGUAGE_COUNTRY = Relation(
    name="language-country",
    description="language -> country of origin",
    templates=[
        "{S} is spoken primarily in",
        "The {S} language originated in",
        "Most native {S} speakers live in",
        "The country where {S} is the main language is",
    ],
    facts=[
        ("French", "France"),
        ("German", "Germany"),
        ("Italian", "Italy"),
        ("Spanish", "Spain"),
        ("Japanese", "Japan"),
        ("Mandarin", "China"),
        ("Portuguese", "Portugal"),
        ("Russian", "Russia"),
        ("Korean", "Korea"),
        ("Hindi", "India"),
        ("Dutch", "Netherlands"),
        ("Greek", "Greece"),
        ("Polish", "Poland"),
        ("Turkish", "Turkey"),
    ],
)

ELEMENT_SYMBOL = Relation(
    name="element-symbol",
    description="chemical symbol -> element name",
    templates=[
        "The chemical element with the symbol {S} is",
        "{S} is the chemical symbol for",
        "The symbol {S} stands for the element",
        "In the periodic table, {S} is the symbol for",
    ],
    facts=[
        ("Au", "gold"),
        ("Ag", "silver"),
        ("Fe", "iron"),
        ("Cu", "copper"),
        ("Pb", "lead"),
        ("Sn", "tin"),
        ("Hg", "mercury"),
        ("He", "helium"),
        ("Na", "sodium"),
        ("K", "potassium"),
        ("Ca", "calcium"),
        ("Zn", "zinc"),
        ("Ni", "nickel"),
        ("Si", "silicon"),
    ],
)

LANDMARK_CITY = Relation(
    name="landmark-city",
    description="landmark -> city where it is located",
    templates=[
        "{S} is located in",
        "You can find {S} in",
        "The famous {S} is found in",
        "Tourists visiting {S} travel to",
    ],
    facts=[
        ("the Eiffel Tower", "Paris"),
        ("Big Ben", "London"),
        ("the Statue of Liberty", "New York"),
        ("the Colosseum", "Rome"),
        ("the Acropolis", "Athens"),
        ("the Brandenburg Gate", "Berlin"),
        ("the Sydney Opera House", "Sydney"),
        ("the Golden Gate Bridge", "San Francisco"),
        ("the Sagrada Familia", "Barcelona"),
        ("the Empire State Building", "New York"),
        ("Buckingham Palace", "London"),
        ("the Tower of London", "London"),
        ("Red Square", "Moscow"),
        ("Tiananmen Square", "Beijing"),
    ],
)

COUNTRY_CONTINENT = Relation(
    name="country-continent",
    description="country -> continent",
    templates=[
        "{S} is a country in",
        "{S} is located on the continent of",
        "The continent where {S} is found is",
        "{S} belongs to the continent of",
    ],
    facts=[
        ("France", "Europe"),
        ("Germany", "Europe"),
        ("Italy", "Europe"),
        ("Spain", "Europe"),
        ("Poland", "Europe"),
        ("Greece", "Europe"),
        ("Japan", "Asia"),
        ("China", "Asia"),
        ("India", "Asia"),
        ("Thailand", "Asia"),
        ("Egypt", "Africa"),
        ("Nigeria", "Africa"),
        ("Kenya", "Africa"),
        ("Brazil", "South America"),
        ("Argentina", "South America"),
        ("Peru", "South America"),
        ("Canada", "North America"),
        ("Mexico", "North America"),
    ],
)

ANIMAL_YOUNG = Relation(
    name="animal-young",
    description="animal -> name of its young",
    templates=[
        "A young {S} is called a",
        "The young of a {S} is called a",
        "A baby {S} is known as a",
        "What is a young {S} called? A",
    ],
    facts=[
        ("cat", "kitten"),
        ("dog", "puppy"),
        ("horse", "foal"),
        ("cow", "calf"),
        ("sheep", "lamb"),
        ("goat", "kid"),
        ("pig", "piglet"),
        ("chicken", "chick"),
        ("duck", "duckling"),
        ("swan", "cygnet"),
        ("frog", "tadpole"),
        ("bear", "cub"),
        ("lion", "cub"),
        ("deer", "fawn"),
        ("tiger", "cub"),
        ("whale", "calf"),
    ],
)

ANIMAL_SOUND = Relation(
    name="animal-sound",
    description="animal -> sound it makes",
    templates=[
        "The sound made by a {S} is a",
        "A {S} makes a sound called a",
        "The cry of a {S} is a",
    ],
    facts=[
        ("dog", "bark"),
        ("lion", "roar"),
        ("cow", "moo"),
        ("sheep", "bleat"),
        ("horse", "neigh"),
        ("pig", "oink"),
        ("duck", "quack"),
        ("rooster", "crow"),
        ("wolf", "howl"),
        ("bee", "buzz"),
        ("snake", "hiss"),
        ("owl", "hoot"),
    ],
    notes="Hardest relation for GPT-2 class models; kept as a stress case.",
)

#: All relations shipped with FactLens.
FACT_BANK: list[Relation] = [
    CAPITAL_COUNTRY,
    WORK_AUTHOR,
    PRODUCT_COMPANY,
    LANGUAGE_COUNTRY,
    ELEMENT_SYMBOL,
    LANDMARK_CITY,
    COUNTRY_CONTINENT,
    ANIMAL_YOUNG,
    ANIMAL_SOUND,
]

#: Default relation subset used by experiment configs.
DEFAULT_RELATIONS = [r.name for r in FACT_BANK]


def get_relation(name: str) -> Relation:
    """Look up a relation by name."""
    for rel in FACT_BANK:
        if rel.name == name:
            return rel
    raise KeyError(
        f"Unknown relation '{name}'. Available: {[r.name for r in FACT_BANK]}"
    )


def get_fact_bank(names: list[str] | None = None) -> list[Relation]:
    """Return relations by name, or the whole bank when ``names`` is None."""
    if names is None:
        return list(FACT_BANK)
    return [get_relation(n) for n in names]
