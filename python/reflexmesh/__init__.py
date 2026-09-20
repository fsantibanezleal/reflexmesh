"""Typed, outcome-grounded software control.

Heavy learning and server dependencies are loaded only by their optional modules.
"""

from .contracts import Candidate, Decision, Observation, Outcome

__version__ = "0.1.0"
__all__ = ["Candidate", "Decision", "Observation", "Outcome", "__version__"]
