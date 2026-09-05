"""Small policy overrides for numerical route coverage."""

from dataclasses import dataclass

from clifra.core._kernel.planning.policy import DEFAULT_PLANNING_POLICY, PolicyEvaluation


@dataclass(frozen=True)
class PreferRoute:
    family: str
    route: str

    def evaluate(self, candidate):
        if (candidate.family, candidate.route) == (self.family, self.route):
            return PolicyEvaluation(-100.0, "test_override")
        return DEFAULT_PLANNING_POLICY.evaluate(candidate)
