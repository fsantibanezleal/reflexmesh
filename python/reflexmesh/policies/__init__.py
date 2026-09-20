from .classical import CandidateScorer, LinUCBPolicy
from .control import (
    AsyncPlannerPolicy,
    DeferralPolicy,
    DirectPlannerPolicy,
    LookaheadPolicy,
    Metacontroller,
)
from .recurrent import RecurrentPolicy
from .rules import BehaviorTreePolicy, GuardedFSMPolicy

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
