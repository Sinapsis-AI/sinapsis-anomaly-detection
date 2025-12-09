# -*- coding: utf-8 -*-
from typing import Any

from pydantic import BaseModel
from sinapsis_core.utils.env_var_keys import EnvVarEntry, doc_str, return_docs_for_vars


class _AnomalibEnvVars(BaseModel):
    """Env vars for Sinapsis Anomalib."""

    ANOMALIB_ROOT_DIR: EnvVarEntry = EnvVarEntry(
        var_name="ANOMALIB_ROOT_DIR",
        default_value="artifacts",
        allowed_values=None,
        description="The base directory for all Anomalib-related artifacts, such as training and exported models.",
    )


AnomalibEnvVars = _AnomalibEnvVars()

doc_str = return_docs_for_vars(AnomalibEnvVars, docs=doc_str, string_for_doc="""Anomalib env vars available: \n""")
__doc__ = doc_str


def __getattr__(name: str) -> Any:
    """To use as an import, when updating the value is not important."""
    if name in AnomalibEnvVars.model_fields:
        return AnomalibEnvVars.model_fields[name].default.value

    raise AttributeError(f"Agent does not have `{name}` env var")


_all__ = (*list(AnomalibEnvVars.model_fields.keys()), "AnomalibEnvVars")
