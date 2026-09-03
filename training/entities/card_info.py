"""Card identity data transfer object."""

from dataclasses import dataclass


@dataclass
class CardInfo:
    """Minimal card identity returned by the scanner.

    Attributes:
        name: Card name (e.g. ``"Lightning Bolt"``).
        setCode: Three-letter set code (e.g. ``"m11"``).
        number: Collector number within the set (e.g. ``"152"``).
        language: Two-letter language code (e.g. ``"en"``).
    """

    name: str
    setCode: str
    number: str
    language: str
