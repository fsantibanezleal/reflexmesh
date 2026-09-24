use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

fn workspace() -> String {
    "default".into()
}
fn normal_capacity() -> usize {
    256
}
fn control_capacity() -> usize {
    32
}
fn event_bytes() -> usize {
    65536
}
fn resource_limit() -> usize {
    4096
}
fn event_limit() -> usize {
    65536
}
fn journal_limit() -> usize {
    100000
}
fn action_resources() -> usize {
    64
}
fn ttl() -> u64 {
    30000
}
fn object() -> Value {
    serde_json::json!({})
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub workspace_id: String,
    pub capabilities: BTreeSet<String>,
    pub trusted_sources: BTreeSet<String>,
    pub normal_capacity: usize,
    pub control_capacity: usize,
    pub max_event_bytes: usize,
    pub max_resources: usize,
    pub max_intents: usize,
    pub max_events: usize,
    pub max_journal_entries: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            workspace_id: workspace(),
            capabilities: BTreeSet::new(),
            trusted_sources: BTreeSet::from(["runtime".into(), "verifier".into()]),
            normal_capacity: normal_capacity(),
            control_capacity: control_capacity(),
            max_event_bytes: event_bytes(),
            max_resources: resource_limit(),
            max_intents: resource_limit(),
            max_events: event_limit(),
            max_journal_entries: journal_limit(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ActionSpec {
    pub action_id: String,
    #[serde(default = "one")]
    pub version: u64,
    #[serde(default)]
    pub required_capabilities: BTreeSet<String>,
    #[serde(default)]
    pub allow_write: bool,
    #[serde(default = "action_resources")]
    pub max_resources: usize,
}
fn one() -> u64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ResourceRef {
    pub resource_id: String,
    pub expected_revision: u64,
    pub access: Access,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Intent {
    pub intent_id: String,
    pub action_id: String,
    #[serde(default = "one")]
    pub action_version: u64,
    #[serde(default)]
    pub resources: Vec<ResourceRef>,
    #[serde(default)]
    pub idempotency_key: Option<String>,
    #[serde(default = "ttl")]
    pub ttl_ms: u64,
    #[serde(default = "object")]
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Accepted,
    Running,
    CancellationRequested,
    CompletedVerified,
    FailedVerified,
    Cancelled,
    Expired,
    Invalidated,
    EffectUnknown,
}
impl Status {
    pub fn terminal(&self) -> bool {
        matches!(
            self,
            Self::CompletedVerified
                | Self::FailedVerified
                | Self::Cancelled
                | Self::Expired
                | Self::Invalidated
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IntentRecord {
    pub intent: Intent,
    pub status: Status,
    pub accepted_ms: u64,
    pub expires_ms: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub cancellation_requested: bool,
    pub outcome: Option<Outcome>,
    pub invalidation_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    CompletedVerified,
    FailedVerified,
    CancelledVerified,
    EffectUnknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub status: OutcomeStatus,
    #[serde(default)]
    pub changed_resources: BTreeSet<String>,
    #[serde(default)]
    pub resource_fingerprints: BTreeMap<String, String>,
    #[serde(default = "object")]
    pub evidence: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResourceState {
    pub revision: u64,
    pub fingerprint: String,
    pub observed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventPayload {
    ResourceObserved {
        resource_id: String,
        fingerprint: String,
    },
    IntentCancel {
        intent_id: String,
    },
    CapabilityRevoke {
        capability: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    pub event_id: String,
    pub source_id: String,
    pub source_sequence: u64,
    #[serde(flatten)]
    pub payload: EventPayload,
}
impl Event {
    pub fn control(&self) -> bool {
        !matches!(self.payload, EventPayload::ResourceObserved { .. })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct State {
    pub actions: BTreeMap<String, ActionSpec>,
    pub capabilities: BTreeSet<String>,
    pub resources: BTreeMap<String, ResourceState>,
    pub intents: BTreeMap<String, IntentRecord>,
    pub operation_keys: BTreeMap<String, String>,
    pub source_sequences: BTreeMap<String, u64>,
    pub event_hashes: BTreeMap<String, String>,
    pub normal_queue: VecDeque<Event>,
    pub control_queue: VecDeque<Event>,
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Command {
    RegisterAction { spec: ActionSpec },
    Observe { event: Event },
    Submit { intent: Intent },
    Begin { intent_id: String },
    Finish { intent_id: String, outcome: Outcome },
    Reconcile { intent_id: String, outcome: Outcome },
    Cancel { intent_id: String },
    Revoke { capability: String },
    Expire,
    Enqueue { event: Event, control: bool },
    Drain { limit: usize },
    Recover,
}
impl Command {
    pub fn critical(&self) -> bool {
        matches!(
            self,
            Self::Finish { .. }
                | Self::Reconcile { .. }
                | Self::Cancel { .. }
                | Self::Revoke { .. }
                | Self::Expire
                | Self::Recover
                | Self::Enqueue { control: true, .. }
        )
    }
}
