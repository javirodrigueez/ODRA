from typing import TYPE_CHECKING

from .state import SessionState, Stage, SubStage

if TYPE_CHECKING:
  from .graph import TherapyGraphBuilder

__all__ = ["TherapyGraphBuilder", "SessionState", "Stage", "SubStage"]


def __getattr__(name: str):
  if name == "TherapyGraphBuilder":
    from .graph import TherapyGraphBuilder
    return TherapyGraphBuilder
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
