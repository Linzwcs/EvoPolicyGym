"""A valid no-placement starting Policy for Treant's Forest."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return {"placements": []}


def make_policy(context: PolicyContext) -> BaselinePolicy:
    del context
    return BaselinePolicy()
