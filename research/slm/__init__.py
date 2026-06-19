# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""TinyStories small-language-model research harness for clifra."""

from .config import SLMConfig
from .model import CliffordTinyStoriesLM, SLMOutput, build_algebra

__all__ = [
    "SLMConfig",
    "SLMOutput",
    "CliffordTinyStoriesLM",
    "build_algebra",
]
