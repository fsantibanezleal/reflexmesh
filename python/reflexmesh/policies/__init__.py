"""Lazy public policy registry; base rule/native serving needs no training stack."""

from importlib import import_module

_MODULES = {
    "CandidateScorer": "classical",
    "LinUCBPolicy": "classical",
    "AsyncPlannerPolicy": "planning",
    "DirectPlannerPolicy": "planning",
    "DeferralPolicy": "control",
    "LookaheadPolicy": "control",
    "Metacontroller": "control",
    "RecurrentPolicy": "recurrent",
    "BehaviorTreePolicy": "rules",
    "GuardedFSMPolicy": "rules",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module("." + _MODULES[name], __name__), name)
    globals()[name] = value
    return value


METHODS = {
    "M01": "Guarded finite-state policy",
    "M02": "Reactive behavior tree",
    "M03": "Calibrated logistic candidate scorer",
    "M04": "Calibrated XGBoost scorer",
    "M05": "GRU behavior policy",
    "M06": "LinUCB routing",
    "M07": "Masked PPO",
    "M08": "Learned-transition bounded lookahead",
    "M09": "Direct structured LLM",
    "M10": "Asynchronous FSM plus LLM",
    "M11": "Learned deferral",
    "M12": "Persistent outcome-aware metacontroller",
}

__all__ = [
    "METHODS",
    "AsyncPlannerPolicy",
    "BehaviorTreePolicy",
    "CandidateScorer",
    "DeferralPolicy",
    "DirectPlannerPolicy",
    "GuardedFSMPolicy",
    "LinUCBPolicy",
    "LookaheadPolicy",
    "Metacontroller",
    "RecurrentPolicy",
]
