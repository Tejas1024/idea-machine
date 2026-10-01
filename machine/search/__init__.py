"""Searches everything he already taught before anything is asked (the #129 rule, L8b)."""
# answered("where does the hedge go?") -> FOUND + citations, or NOT FOUND + the #129 sentence.
# CLI: python -m machine.search "question" · the exam: python -m machine.search.exam
from machine.search.rank import Answer, Hit, answered, search

__all__ = ["Answer", "Hit", "answered", "search"]
