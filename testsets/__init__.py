"""Builders of corrupted test sets from public speech, driven by YAML recipes.

This package lives outside `src/` on purpose: it is a tool for making benchmark data,
not part of the installed paribus. Its output is an ordinary paribus manifest, so the
test sets it builds are read like any user's data.

Your own operations need only what is exported here: subclass `Operation`, override
`apply`, and register it with `register_op`; see the README.
"""

from testsets.ops import Clip, Samples
from testsets.recipe import OpContext, Operation, Range, register_op

__all__ = ["Clip", "OpContext", "Operation", "Range", "Samples", "register_op"]
