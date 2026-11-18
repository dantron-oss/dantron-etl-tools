"""The names this module used to define, re-exported from the modules that now hold them.

The loader is now ``loader.py``, the classes a loaded flow is made of are in ``spec.py``, and rule
references are resolved in ``rules.py``. ``Pipeline``, ``Step`` and ``read_pipeline`` are the new
names of ``PipelineSpec``, ``StepSpec`` and ``read_pipeline_file``. This module only keeps old
imports working until the client code is moved over.
"""

# compatibility: removed in the names stage
from .errors import PipelineError
from .loader import load_pipeline, read_pipeline
from .registry import REGISTRY
from .spec import Group, Lane, Pipeline, PortRef, Step

PipelineSpec = Pipeline  # compatibility: removed in the names stage
StepSpec = Step  # compatibility: removed in the names stage
read_pipeline_file = read_pipeline  # compatibility: removed in the names stage

__all__ = [
    "REGISTRY",
    "Group",
    "Lane",
    "Pipeline",
    "PipelineError",
    "PipelineSpec",
    "PortRef",
    "Step",
    "StepSpec",
    "load_pipeline",
    "read_pipeline",
    "read_pipeline_file",
]
