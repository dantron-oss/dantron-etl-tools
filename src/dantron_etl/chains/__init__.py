"""Chains: several pipelines run in order, each after the ones whose files and tables it reads.

A chain is a YAML file in a project's ``chains`` folder. It names the pipelines, what each runs
after, and parameters shared by all of them. Not imported by ``import`` of the library itself.

- :func:`load_chain` reads a chain, loads its pipelines and checks the order against what they read
  and write. Nothing runs.
- :func:`run_chain` runs a checked chain for real, side by side where the order allows.
- :func:`run_member` runs one pipeline of a chain, for a scheduler that runs each as its own task.
- :func:`suggest_chain` writes a first chain file from what a set of pipelines read and write.
- :func:`chain_runs` lists a chain's saved runs.

Why: docs/design/chains.md
"""

from .chain_file import CHAIN_KEYS, DEFAULT_MAX_PARALLEL, MEMBER_KEYS, ChainError
from .plan import ChainPlan, Handoff, MemberPlan, load_chain
from .running import ChainRun, MemberResult, chain_runs, run_chain, run_member
from .suggest import suggest_chain

__all__ = [
    "CHAIN_KEYS",
    "DEFAULT_MAX_PARALLEL",
    "MEMBER_KEYS",
    "ChainError",
    "ChainPlan",
    "ChainRun",
    "Handoff",
    "MemberPlan",
    "MemberResult",
    "chain_runs",
    "load_chain",
    "run_chain",
    "run_member",
    "suggest_chain",
]
