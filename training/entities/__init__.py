"""Domain entities for the MTG card scanner."""

from entities.card_info import CardInfo
from entities.scan_result import MatchConfidence, ScanResult

__all__ = ["CardInfo", "MatchConfidence", "ScanResult"]
