use crate::contracts::*;
use crate::core::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

fn run(core: &mut Core, command: Command, tick: u64) -> Value {
    core.transact(command, tick).unwrap()
}
fn config() -> Config {
    Config {
        capabilities: BTreeSet::from(["files.write".into()]),
        ..Config::default()
    }
}
fn spec() -> ActionSpec {
    ActionSpec {
        action_id: "write".into(),
        version: 1,
        required_capabilities: BTreeSet::from(["files.write".into()]),
        allow_write: true,
        max_resources: 4,
    }
}
fn observed(seq: u64, fingerprint: &str) -> Event {
    Event {
        event_id: format!("e{seq}"),
        source_id: "runtime".into(),
        source_sequence: seq,
        payload: EventPayload::ResourceObserved {
            resource_id: "file:a".into(),
            fingerprint: fingerprint.into(),
        },
    }
}
fn intent(key: &str) -> Intent {
    Intent {
        intent_id: key.into(),
        action_id: "write".into(),
        action_version: 1,
        resources: vec![ResourceRef {
            resource_id: "file:a".into(),
            expected_revision: 0,
            access: Access::Write,
        }],
        idempotency_key: Some(format!("operation:{key}")),
        ttl_ms: 100,
        arguments: json!({"content":"hello"}),
    }
}
fn outcome(status: OutcomeStatus) -> Outcome {
    Outcome {
        status,
        changed_resources: BTreeSet::new(),
        resource_fingerprints: BTreeMap::new(),
        evidence: json!({"verifier":"test-state","receipt":"r1"}),
    }
}
fn ready() -> Core {
    let mut core = Core::new(config()).unwrap();
    run(&mut core, Command::RegisterAction { spec: spec() }, 0);
    run(
        &mut core,
        Command::Observe {
            event: observed(1, "before"),
        },
        1,
    );
    core
}

#[test]
fn immutable_registration_and_capabilities_are_rechecked() {
    let mut core = ready();
    let mut changed = spec();
    changed.required_capabilities.clear();
    assert!(core
        .transact(Command::RegisterAction { spec: changed }, 2)
        .unwrap_err()
        .contains("immutable_action"));
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    run(
        &mut core,
        Command::Revoke {
            capability: "files.write".into(),
        },
        3,
    );
    assert_eq!(core.state.intents["a"].status, Status::Cancelled);
    assert!(core
        .transact(
            Command::Begin {
                intent_id: "a".into()
            },
            4
        )
        .is_err());
    assert!(core
        .transact(
            Command::Submit {
                intent: intent("b")
            },
            4
        )
        .unwrap_err()
        .contains("capability_denied"));
}

#[test]
fn stale_events_and_conflicting_duplicates_cannot_change_state() {
    let mut core = ready();
    let duplicate = run(
        &mut core,
        Command::Observe {
            event: observed(1, "before"),
        },
        2,
    );
    assert_eq!(duplicate["duplicate"], true);
    let prior = core.snapshot();
    assert!(core
        .transact(
            Command::Observe {
                event: observed(1, "other")
            },
            3
        )
        .unwrap_err()
        .contains("event_collision"));
    let mut old = observed(1, "other");
    old.event_id = "new-id".into();
    assert!(core
        .transact(Command::Observe { event: old }, 3)
        .unwrap_err()
        .contains("stale_sequence"));
    assert_eq!(core.snapshot(), prior);
    let mut untrusted = observed(2, "other");
    untrusted.source_id = "tool-output".into();
    assert!(core
        .transact(Command::Observe { event: untrusted }, 3)
        .unwrap_err()
        .contains("untrusted_source"));
}

#[test]
fn same_fingerprint_preserves_revision_changed_fingerprint_invalidates_pending() {
    let mut core = ready();
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    run(
        &mut core,
        Command::Observe {
            event: observed(2, "before"),
        },
        3,
    );
    assert_eq!(core.state.resources["file:a"].revision, 0);
    run(
        &mut core,
        Command::Observe {
            event: observed(3, "external-change"),
        },
        4,
    );
    assert_eq!(core.state.resources["file:a"].revision, 1);
    assert_eq!(core.state.intents["a"].status, Status::Invalidated);
    assert!(core
        .transact(
            Command::Begin {
                intent_id: "a".into()
            },
            5
        )
        .is_err());
    assert!(core
        .transact(
            Command::Submit {
                intent: intent("b")
            },
            5
        )
        .unwrap_err()
        .contains("stale_revision"));
}

#[test]
fn read_leases_share_but_write_lease_conflicts() {
    let mut core = ready();
    let mut a = intent("a");
    a.resources[0].access = Access::Read;
    let mut b = intent("b");
    b.resources[0].access = Access::Read;
    run(&mut core, Command::Submit { intent: a }, 2);
    run(&mut core, Command::Submit { intent: b }, 3);
    assert!(core
        .transact(
            Command::Submit {
                intent: intent("c")
            },
            4
        )
        .unwrap_err()
        .contains("lease_conflict"));
    run(
        &mut core,
        Command::Cancel {
            intent_id: "a".into(),
        },
        5,
    );
    run(
        &mut core,
        Command::Cancel {
            intent_id: "b".into(),
        },
        6,
    );
    run(
        &mut core,
        Command::Submit {
            intent: intent("c"),
        },
        7,
    );
}

#[test]
fn duplicate_intents_never_provide_a_second_dispatch() {
    let mut core = ready();
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    let duplicate = run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        3,
    );
    assert_eq!(duplicate["duplicate"], true);
    let mut alias = intent("a");
    alias.intent_id = "alias".into();
    let duplicate = run(&mut core, Command::Submit { intent: alias }, 3);
    assert_eq!(duplicate["record"]["intent"]["intent_id"], "a");
    run(
        &mut core,
        Command::Begin {
            intent_id: "a".into(),
        },
        4,
    );
    assert!(core
        .transact(
            Command::Begin {
                intent_id: "a".into()
            },
            5
        )
        .unwrap_err()
        .contains("not_dispatchable"));
    let mut changed = intent("a");
    changed.arguments = json!({"content":"different"});
    assert!(core
        .transact(Command::Submit { intent: changed }, 5)
        .unwrap_err()
        .contains("intent_collision"));
}

#[test]
fn cancellation_after_dispatch_does_not_release_unknown_write_lease() {
    let mut core = ready();
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    run(
        &mut core,
        Command::Begin {
            intent_id: "a".into(),
        },
        3,
    );
    run(
        &mut core,
        Command::Cancel {
            intent_id: "a".into(),
        },
        4,
    );
    assert_eq!(
        core.state.intents["a"].status,
        Status::CancellationRequested
    );
    run(
        &mut core,
        Command::Finish {
            intent_id: "a".into(),
            outcome: outcome(OutcomeStatus::EffectUnknown),
        },
        5,
    );
    assert!(core
        .transact(
            Command::Submit {
                intent: intent("b")
            },
            200
        )
        .unwrap_err()
        .contains("lease_conflict"));
    assert!(core
        .transact(
            Command::Finish {
                intent_id: "a".into(),
                outcome: outcome(OutcomeStatus::FailedVerified)
            },
            201
        )
        .is_err());
    run(
        &mut core,
        Command::Reconcile {
            intent_id: "a".into(),
            outcome: outcome(OutcomeStatus::FailedVerified),
        },
        202,
    );
    run(
        &mut core,
        Command::Submit {
            intent: intent("b"),
        },
        203,
    );
}

#[test]
fn verified_effects_advance_authoritative_revision_once() {
    let mut core = ready();
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    assert!(core
        .transact(
            Command::Finish {
                intent_id: "a".into(),
                outcome: outcome(OutcomeStatus::CompletedVerified)
            },
            3
        )
        .is_err());
    run(
        &mut core,
        Command::Begin {
            intent_id: "a".into(),
        },
        3,
    );
    let mut done = outcome(OutcomeStatus::CompletedVerified);
    done.changed_resources.insert("file:a".into());
    done.resource_fingerprints
        .insert("file:a".into(), "after".into());
    run(
        &mut core,
        Command::Finish {
            intent_id: "a".into(),
            outcome: done.clone(),
        },
        4,
    );
    run(
        &mut core,
        Command::Finish {
            intent_id: "a".into(),
            outcome: done,
        },
        5,
    );
    run(
        &mut core,
        Command::Observe {
            event: observed(2, "after"),
        },
        6,
    );
    assert_eq!(core.state.resources["file:a"].revision, 1);
}

#[test]
fn rejected_effect_scope_is_atomic_and_evidence_is_required() {
    let mut core = ready();
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    run(
        &mut core,
        Command::Begin {
            intent_id: "a".into(),
        },
        3,
    );
    let before = core.snapshot();
    let mut bad = outcome(OutcomeStatus::CompletedVerified);
    bad.changed_resources = BTreeSet::from(["file:a".into(), "file:z".into()]);
    assert!(core
        .transact(
            Command::Finish {
                intent_id: "a".into(),
                outcome: bad
            },
            4
        )
        .unwrap_err()
        .contains("effect_scope"));
    assert_eq!(core.snapshot(), before);
    let mut bad = outcome(OutcomeStatus::CompletedVerified);
    bad.evidence = json!({});
    assert!(core
        .transact(
            Command::Finish {
                intent_id: "a".into(),
                outcome: bad
            },
            4
        )
        .unwrap_err()
        .contains("missing_evidence"));
}

#[test]
fn expiry_blocks_dispatch_and_releases_only_unstarted_work() {
    let mut core = ready();
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    assert!(core
        .transact(
            Command::Begin {
                intent_id: "a".into()
            },
            102
        )
        .unwrap_err()
        .contains("expired"));
    run(&mut core, Command::Expire, 102);
    assert_eq!(core.state.intents["a"].status, Status::Expired);
    run(
        &mut core,
        Command::Submit {
            intent: intent("b"),
        },
        103,
    );
    run(
        &mut core,
        Command::Begin {
            intent_id: "b".into(),
        },
        104,
    );
    run(&mut core, Command::Expire, 10000);
    assert_eq!(core.state.intents["b"].status, Status::Running);
}

#[test]
fn control_queue_is_reserved_and_drained_before_saturated_normal_queue() {
    let mut cfg = config();
    cfg.normal_capacity = 2;
    cfg.control_capacity = 1;
    let mut core = Core::new(cfg).unwrap();
    run(&mut core, Command::RegisterAction { spec: spec() }, 0);
    run(
        &mut core,
        Command::Observe {
            event: observed(1, "before"),
        },
        1,
    );
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    for seq in 2..=3 {
        run(
            &mut core,
            Command::Enqueue {
                event: observed(seq, "before"),
                control: false,
            },
            3,
        );
    }
    assert!(core
        .transact(
            Command::Enqueue {
                event: observed(4, "before"),
                control: false
            },
            4
        )
        .unwrap_err()
        .contains("queue_full"));
    let stop = Event {
        event_id: "stop".into(),
        source_id: "verifier".into(),
        source_sequence: 1,
        payload: EventPayload::IntentCancel {
            intent_id: "a".into(),
        },
    };
    run(
        &mut core,
        Command::Enqueue {
            event: stop.clone(),
            control: true,
        },
        4,
    );
    assert!(core
        .transact(
            Command::Enqueue {
                event: stop,
                control: true
            },
            5
        )
        .unwrap_err()
        .contains("queue_full"));
    let batch = run(&mut core, Command::Drain { limit: 1 }, 5);
    assert_eq!(batch[0]["event_id"], "stop");
    assert_eq!(core.state.intents["a"].status, Status::Cancelled);
    assert_eq!(core.state.normal_queue.len(), 2);
    assert!(core.transact(Command::Drain { limit: 65 }, 6).is_err());
}

#[test]
fn replay_reconstructs_state_and_detects_tampering_without_dispatch() {
    let mut core = ready();
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    run(
        &mut core,
        Command::Begin {
            intent_id: "a".into(),
        },
        3,
    );
    let reconstructed = Core::replay(core.journal.clone()).unwrap();
    assert_eq!(core.snapshot(), reconstructed.snapshot());
    let mut corrupt = core.journal.clone();
    corrupt.entries[0].tick_ms = 55;
    assert!(Core::replay(corrupt).is_err());
    let mut recovered = Core::replay(core.journal.clone()).unwrap();
    let result = run(&mut recovered, Command::Recover, 4);
    assert_eq!(result["effects_executed"], 0);
    assert_eq!(recovered.state.intents["a"].status, Status::EffectUnknown);
    assert!(recovered
        .transact(
            Command::Submit {
                intent: intent("b")
            },
            5
        )
        .unwrap_err()
        .contains("lease_conflict"));
}

#[test]
fn checkpoint_retains_duplicate_ledger_and_replay_equivalence() {
    let mut core = ready();
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    run(
        &mut core,
        Command::Cancel {
            intent_id: "a".into(),
        },
        3,
    );
    assert!(core.checkpoint("wrong").is_err());
    let tip = core.tip().to_owned();
    core.checkpoint(&tip).unwrap();
    assert!(core.journal.entries.is_empty());
    run(
        &mut core,
        Command::Submit {
            intent: intent("b"),
        },
        4,
    );
    let replayed = Core::replay(core.journal.clone()).unwrap();
    assert_eq!(core.snapshot(), replayed.snapshot());
    assert_eq!(
        run(
            &mut core,
            Command::Submit {
                intent: intent("a")
            },
            5
        )["duplicate"],
        true
    );
}

#[test]
fn journal_backpressure_keeps_reserved_space_for_cancel_and_checkpoint() {
    let mut cfg = config();
    cfg.max_journal_entries = 32;
    let mut core = Core::new(cfg).unwrap();
    run(&mut core, Command::RegisterAction { spec: spec() }, 0);
    run(
        &mut core,
        Command::Observe {
            event: observed(1, "before"),
        },
        1,
    );
    run(
        &mut core,
        Command::Submit {
            intent: intent("a"),
        },
        2,
    );
    let mut blocked = false;
    for seq in 2..100 {
        if core
            .transact(
                Command::Observe {
                    event: observed(seq, "before"),
                },
                seq + 1,
            )
            .is_err()
        {
            blocked = true;
            break;
        }
    }
    assert!(blocked);
    run(
        &mut core,
        Command::Cancel {
            intent_id: "a".into(),
        },
        200,
    );
    assert_eq!(core.state.intents["a"].status, Status::Cancelled);
    let tip = core.tip().to_owned();
    core.checkpoint(&tip).unwrap();
    run(
        &mut core,
        Command::Submit {
            intent: intent("b"),
        },
        201,
    );
}

#[test]
fn wire_schema_flattens_typed_events_and_rejects_invalid_enums() {
    let e: Event = serde_json::from_value(
        json!({"event_id":"a","source_id":"runtime","source_sequence":1,
        "kind":"resource_observed","resource_id":"file:a","fingerprint":"x"}),
    )
    .unwrap();
    assert!(!e.control());
    assert!(serde_json::from_value::<Event>(
        json!({"event_id":"a","source_id":"runtime","source_sequence":1,
        "kind":"shell_execute","command":"echo bad"})
    )
    .is_err());
}

#[test]
fn revision_and_clock_regression_do_not_wrap() {
    let mut core = ready();
    assert!(core
        .transact(Command::Expire, 0)
        .unwrap_err()
        .contains("clock_regression"));
    core.state.resources.get_mut("file:a").unwrap().revision = u64::MAX;
    assert!(core
        .transact(
            Command::Observe {
                event: observed(2, "new")
            },
            2
        )
        .is_err());
    assert_eq!(core.state.resources["file:a"].revision, u64::MAX);
}
