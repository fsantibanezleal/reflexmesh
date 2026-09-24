use crate::contracts::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub type Result<T> = std::result::Result<T, String>;
fn fail<T>(code: &str, detail: impl AsRef<str>) -> Result<T> {
    Err(format!("{code}: {}", detail.as_ref()))
}
fn id(value: &str) -> Result<()> {
    if value.is_empty() || value.len() > 512 || value.chars().any(char::is_control) {
        fail(
            "invalid_identifier",
            "identifiers must contain 1..512 non-control bytes",
        )
    } else {
        Ok(())
    }
}
pub fn digest<T: Serialize>(value: &T) -> Result<String> {
    let encoded = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}
fn value<T: Serialize>(data: &T) -> Result<Value> {
    serde_json::to_value(data).map_err(|e| e.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub sequence: u64,
    pub tick_ms: u64,
    pub previous_hash: String,
    pub command: Command,
    pub result_hash: String,
    pub hash: String,
}
impl Entry {
    fn computed_hash(&self) -> Result<String> {
        digest(&(
            self.sequence,
            self.tick_ms,
            &self.previous_hash,
            &self.command,
            &self.result_hash,
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Journal {
    pub schema_version: u32,
    pub config: Config,
    pub checkpoint_state: State,
    pub checkpoint_sequence: u64,
    pub checkpoint_tick_ms: u64,
    pub checkpoint_parent_hash: String,
    pub checkpoint_hash: String,
    pub entries: Vec<Entry>,
}

pub struct Core {
    pub config: Config,
    pub state: State,
    pub journal: Journal,
    pub tick_ms: u64,
}

impl Core {
    pub fn new(config: Config) -> Result<Self> {
        id(&config.workspace_id)?;
        if config.normal_capacity == 0
            || config.control_capacity == 0
            || config.normal_capacity > 65536
            || config.control_capacity > 4096
            || config.max_event_bytes < 256
            || config.max_event_bytes > 1048576
            || config.max_resources == 0
            || config.max_intents == 0
            || config.max_events == 0
            || config.max_resources > 100000
            || config.max_intents > 100000
            || config.max_events > 1000000
            || config.max_journal_entries < 32
        {
            return fail("invalid_config", "capacity bounds violated");
        }
        for cap in &config.capabilities {
            id(cap)?;
        }
        for source in &config.trusted_sources {
            id(source)?;
        }
        let state = State {
            capabilities: config.capabilities.clone(),
            ..State::default()
        };
        let checkpoint_hash = digest(&(&config, &state, 0_u64, 0_u64, "genesis"))?;
        let journal = Journal {
            schema_version: 1,
            config: config.clone(),
            checkpoint_state: state.clone(),
            checkpoint_sequence: 0,
            checkpoint_tick_ms: 0,
            checkpoint_parent_hash: "genesis".into(),
            checkpoint_hash,
            entries: vec![],
        };
        Ok(Self {
            config,
            state,
            journal,
            tick_ms: 0,
        })
    }

    pub fn tip(&self) -> &str {
        self.journal
            .entries
            .last()
            .map(|e| e.hash.as_str())
            .unwrap_or(&self.journal.checkpoint_hash)
    }
    pub fn sequence(&self) -> u64 {
        self.journal
            .entries
            .last()
            .map(|e| e.sequence)
            .unwrap_or(self.journal.checkpoint_sequence)
    }

    /// Commit only after all validation succeeds. Rejected commands cannot partially mutate state.
    pub fn transact(&mut self, command: Command, tick_ms: u64) -> Result<Value> {
        if tick_ms < self.tick_ms {
            return fail("clock_regression", "monotonic time moved backwards");
        }
        if serde_json::to_vec(&command)
            .map_err(|e| e.to_string())?
            .len()
            > self.config.max_event_bytes
        {
            return fail("payload_limit", "command exceeds configured byte limit");
        }
        let mut next = self.state.clone();
        let result = apply(&self.config, &mut next, &command, tick_ms)?;
        if next == self.state {
            return Ok(result);
        }
        let active = next
            .intents
            .values()
            .filter(|r| !r.status.terminal())
            .count();
        // Admission leaves room for outcomes/control. The host durably checkpoints before exhaustion.
        let reserve = active.saturating_mul(3).saturating_add(16);
        let capacity = self.config.max_journal_entries;
        let control_drain =
            matches!(command, Command::Drain { .. }) && !self.state.control_queue.is_empty();
        if self.journal.entries.len() >= capacity
            || (!command.critical()
                && !control_drain
                && self.journal.entries.len().saturating_add(reserve) >= capacity)
        {
            return fail(
                "journal_backpressure",
                "persist journal and acknowledge checkpoint before more mutations",
            );
        }
        let mut entry = Entry {
            sequence: self.sequence().checked_add(1).ok_or("sequence_overflow")?,
            tick_ms,
            previous_hash: self.tip().to_owned(),
            command,
            result_hash: digest(&result)?,
            hash: String::new(),
        };
        entry.hash = entry.computed_hash()?;
        self.state = next;
        self.tick_ms = tick_ms;
        self.journal.entries.push(entry);
        Ok(result)
    }

    pub fn snapshot(&self) -> Value {
        json!({"schema_version":1,"workspace_id":self.config.workspace_id,
            "revision":self.state.revision,"tick_ms":self.tick_ms,"journal_tip":self.tip(),
            "journal_sequence":self.sequence(),"state":self.state})
    }

    /// A checkpoint is acknowledged only after the caller has durably stored the current journal.
    pub fn checkpoint(&mut self, acknowledged_tip: &str) -> Result<Value> {
        if acknowledged_tip != self.tip() {
            return fail("checkpoint_mismatch", "acknowledged tip differs");
        }
        let parent = self.tip().to_owned();
        let sequence = self.sequence();
        let hash = digest(&(&self.config, &self.state, sequence, self.tick_ms, &parent))?;
        self.journal = Journal {
            schema_version: 1,
            config: self.config.clone(),
            checkpoint_state: self.state.clone(),
            checkpoint_sequence: sequence,
            checkpoint_tick_ms: self.tick_ms,
            checkpoint_parent_hash: parent,
            checkpoint_hash: hash,
            entries: vec![],
        };
        Ok(json!({"checkpoint_hash":self.journal.checkpoint_hash,"sequence":sequence}))
    }

    pub fn replay(journal: Journal) -> Result<Self> {
        if journal.schema_version != 1 {
            return fail("journal_version", "unsupported schema version");
        }
        let mut core = Self::new(journal.config.clone())?;
        if journal.entries.len() > journal.config.max_journal_entries {
            return fail("journal_limit", "too many entries");
        }
        let expected = digest(&(
            &journal.config,
            &journal.checkpoint_state,
            journal.checkpoint_sequence,
            journal.checkpoint_tick_ms,
            &journal.checkpoint_parent_hash,
        ))?;
        if expected != journal.checkpoint_hash {
            return fail("journal_integrity", "checkpoint digest mismatch");
        }
        validate_state(&journal.config, &journal.checkpoint_state)?;
        core.state = journal.checkpoint_state.clone();
        core.tick_ms = journal.checkpoint_tick_ms;
        let entries = journal.entries.clone();
        core.journal = Journal {
            entries: vec![],
            ..journal
        };
        for expected in entries {
            if Some(expected.sequence) != core.sequence().checked_add(1)
                || expected.previous_hash != core.tip()
                || expected.computed_hash()? != expected.hash
            {
                return fail("journal_integrity", "entry chain mismatch");
            }
            let before = core.journal.entries.len();
            let result = core.transact(expected.command.clone(), expected.tick_ms)?;
            if core.journal.entries.len() != before + 1
                || digest(&result)? != expected.result_hash
                || core.tip() != expected.hash
            {
                return fail(
                    "journal_replay",
                    "recorded transition differs from reducer result",
                );
            }
        }
        Ok(core)
    }
}

fn validate_state(config: &Config, state: &State) -> Result<()> {
    if state.resources.len() > config.max_resources
        || state.intents.len() > config.max_intents
        || state.event_hashes.len() > config.max_events
        || state.normal_queue.len() > config.normal_capacity
        || state.control_queue.len() > config.control_capacity
    {
        return fail("checkpoint_limit", "state exceeds configured limits");
    }
    if !state.capabilities.is_subset(&config.capabilities) {
        return fail(
            "checkpoint_capabilities",
            "checkpoint grants undeclared capabilities",
        );
    }
    for (key, record) in &state.intents {
        if key != &record.intent.intent_id || !state.actions.contains_key(&record.intent.action_id)
        {
            return fail("checkpoint_intent", "inconsistent intent registry");
        }
    }
    Ok(())
}

fn apply(config: &Config, state: &mut State, command: &Command, now: u64) -> Result<Value> {
    match command {
        Command::RegisterAction { spec } => {
            id(&spec.action_id)?;
            if spec.version == 0 || spec.max_resources > 64 {
                return fail(
                    "invalid_action",
                    "version must be positive and max_resources <=64",
                );
            }
            for capability in &spec.required_capabilities {
                id(capability)?;
            }
            if let Some(old) = state.actions.get(&spec.action_id) {
                return if old == spec {
                    value(old)
                } else {
                    fail(
                        "immutable_action",
                        "register a new action ID/version instead of replacing a contract",
                    )
                };
            }
            if state.actions.len() >= config.max_resources {
                return fail("action_limit", "registry full");
            }
            state.actions.insert(spec.action_id.clone(), spec.clone());
            value(spec)
        }
        Command::Observe { event } => observe(config, state, event, now),
        Command::Submit { intent } => submit(config, state, intent, now),
        Command::Begin { intent_id } => begin(state, intent_id, now),
        Command::Finish { intent_id, outcome } => finish(state, intent_id, outcome, now, false),
        Command::Reconcile { intent_id, outcome } => finish(state, intent_id, outcome, now, true),
        Command::Cancel { intent_id } => cancel(state, intent_id, now),
        Command::Revoke { capability } => revoke(state, capability, now),
        Command::Expire => Ok(json!({"expired":expire(state, now)})),
        Command::Enqueue { event, control } => {
            validate_event(config, event)?;
            if event.control() != *control {
                return fail("queue_class", "control flag must match event kind");
            }
            let queue = if *control {
                &mut state.control_queue
            } else {
                &mut state.normal_queue
            };
            let capacity = if *control {
                config.control_capacity
            } else {
                config.normal_capacity
            };
            if queue.len() >= capacity {
                return fail("queue_full", "bounded queue capacity reached");
            }
            queue.push_back(event.clone());
            Ok(json!({"queued":true,"control":control,"depth":queue.len()}))
        }
        Command::Drain { limit } => {
            if *limit == 0 || *limit > 64 {
                return fail("drain_limit", "limit must be 1..64");
            }
            let mut results = Vec::new();
            for _ in 0..*limit {
                let event = state
                    .control_queue
                    .pop_front()
                    .or_else(|| state.normal_queue.pop_front());
                let Some(event) = event else {
                    break;
                };
                // Failed events are dead-lettered as explicit errors without corrupting source sequences.
                let mut after = state.clone();
                match observe(config, &mut after, &event, now) {
                    Ok(result) => {
                        *state = after;
                        results.push(json!({"event_id":event.event_id,"result":result}));
                    }
                    Err(error) => results.push(json!({"event_id":event.event_id,"error":error})),
                }
            }
            Ok(json!(results))
        }
        Command::Recover => {
            let mut unknown = vec![];
            let mut invalidated = vec![];
            for (key, record) in &mut state.intents {
                match record.status {
                    Status::Accepted => {
                        record.status = Status::Invalidated;
                        record.invalidation_reason =
                            Some("process_restart_requires_new_observation".into());
                        record.finished_ms = Some(now);
                        invalidated.push(key.clone());
                    }
                    Status::Running | Status::CancellationRequested => {
                        record.status = Status::EffectUnknown;
                        unknown.push(key.clone());
                    }
                    _ => {}
                }
            }
            state.normal_queue.clear();
            state.control_queue.clear();
            Ok(json!({"unknown":unknown,"invalidated":invalidated,"effects_executed":0}))
        }
    }
}

fn validate_event(config: &Config, event: &Event) -> Result<()> {
    id(&event.event_id)?;
    id(&event.source_id)?;
    if !config.trusted_sources.contains(&event.source_id) {
        return fail("untrusted_source", &event.source_id);
    }
    if event.source_sequence == 0 {
        return fail("invalid_sequence", "source_sequence must start at 1");
    }
    if serde_json::to_vec(event).map_err(|e| e.to_string())?.len() > config.max_event_bytes {
        return fail("payload_limit", "event exceeds configured byte limit");
    }
    Ok(())
}

fn observe(config: &Config, state: &mut State, event: &Event, now: u64) -> Result<Value> {
    validate_event(config, event)?;
    let hash = digest(event)?;
    if let Some(old) = state.event_hashes.get(&event.event_id) {
        return if old == &hash {
            Ok(json!({"duplicate":true,"revision":state.revision}))
        } else {
            fail("event_collision", "event ID reused with different payload")
        };
    }
    if state.event_hashes.len() >= config.max_events {
        return fail(
            "event_limit",
            "deduplication ledger full; rotate workspace with durable history",
        );
    }
    let last = *state.source_sequences.get(&event.source_id).unwrap_or(&0);
    if event.source_sequence <= last {
        return fail("stale_sequence", "source event already passed");
    }
    let gap = event.source_sequence > last + 1;
    let result = match &event.payload {
        EventPayload::ResourceObserved {
            resource_id,
            fingerprint,
        } => {
            id(resource_id)?;
            if fingerprint.len() > 4096 {
                return fail("fingerprint_limit", "fingerprint too long");
            }
            let changed = state
                .resources
                .get(resource_id)
                .is_some_and(|r| r.fingerprint != *fingerprint);
            if !state.resources.contains_key(resource_id)
                && state.resources.len() >= config.max_resources
            {
                return fail("resource_limit", "resource registry full");
            }
            let revision = match state.resources.get(resource_id) {
                Some(old) if changed => old.revision.checked_add(1).ok_or("revision_overflow")?,
                Some(old) => old.revision,
                None => 0,
            };
            state.resources.insert(
                resource_id.clone(),
                ResourceState {
                    revision,
                    fingerprint: fingerprint.clone(),
                    observed_ms: now,
                },
            );
            let mut invalidated = vec![];
            if changed {
                for (key, record) in &mut state.intents {
                    if record.status == Status::Accepted
                        && record
                            .intent
                            .resources
                            .iter()
                            .any(|r| &r.resource_id == resource_id)
                    {
                        record.status = Status::Invalidated;
                        record.invalidation_reason =
                            Some("resource_changed_before_dispatch".into());
                        record.finished_ms = Some(now);
                        invalidated.push(key.clone());
                    }
                }
            }
            json!({"resource_id":resource_id,"resource_revision":revision,"changed":changed,"invalidated":invalidated})
        }
        EventPayload::IntentCancel { intent_id } => cancel(state, intent_id, now)?,
        EventPayload::CapabilityRevoke { capability } => revoke(state, capability, now)?,
    };
    state
        .source_sequences
        .insert(event.source_id.clone(), event.source_sequence);
    state.event_hashes.insert(event.event_id.clone(), hash);
    state.revision = state.revision.checked_add(1).ok_or("revision_overflow")?;
    Ok(json!({"duplicate":false,"sequence_gap":gap,"revision":state.revision,"result":result}))
}

fn same_operation(a: &Intent, b: &Intent) -> bool {
    a.action_id == b.action_id
        && a.action_version == b.action_version
        && a.resources == b.resources
        && a.arguments == b.arguments
        && a.idempotency_key == b.idempotency_key
}
fn operation_key(intent: &Intent) -> Option<String> {
    intent
        .idempotency_key
        .as_ref()
        .map(|key| format!("{}:{}:{}", intent.action_id.len(), intent.action_id, key))
}
fn check_authority(state: &State, intent: &Intent) -> Result<ActionSpec> {
    let spec = state
        .actions
        .get(&intent.action_id)
        .ok_or_else(|| format!("unregistered_action: {}", intent.action_id))?;
    if intent.action_version != spec.version {
        return fail("action_version", "registered action version differs");
    }
    if !spec.required_capabilities.is_subset(&state.capabilities) {
        return fail("capability_denied", "required capability absent/revoked");
    }
    if intent.resources.len() > spec.max_resources {
        return fail("resource_limit", "action resource bound exceeded");
    }
    if !spec.allow_write && intent.resources.iter().any(|r| r.access == Access::Write) {
        return fail("write_denied", "registered action cannot write");
    }
    for resource in &intent.resources {
        let observed = state
            .resources
            .get(&resource.resource_id)
            .ok_or_else(|| format!("unknown_resource: {}", resource.resource_id))?;
        if observed.revision != resource.expected_revision {
            return fail("stale_revision", &resource.resource_id);
        }
    }
    Ok(spec.clone())
}

fn conflicts(a: &Intent, b: &Intent) -> bool {
    a.resources.iter().any(|x| {
        b.resources.iter().any(|y| {
            x.resource_id == y.resource_id
                && (x.access == Access::Write || y.access == Access::Write)
        })
    })
}

fn submit(config: &Config, state: &mut State, intent: &Intent, now: u64) -> Result<Value> {
    id(&intent.intent_id)?;
    id(&intent.action_id)?;
    if let Some(key) = &intent.idempotency_key {
        id(key)?;
    }
    if !intent.arguments.is_object() {
        return fail("invalid_arguments", "arguments must be an object");
    }
    if intent.ttl_ms == 0 || intent.ttl_ms > 86400000 {
        return fail("invalid_ttl", "ttl_ms must be 1..86400000");
    }
    let unique: BTreeSet<_> = intent.resources.iter().map(|r| &r.resource_id).collect();
    if unique.len() != intent.resources.len() {
        return fail("duplicate_resource", "resource references must be unique");
    }
    for resource in &intent.resources {
        id(&resource.resource_id)?;
    }
    if let Some(record) = state.intents.get(&intent.intent_id) {
        return if same_operation(&record.intent, intent) {
            Ok(json!({"duplicate":true,"record":record}))
        } else {
            fail(
                "intent_collision",
                "intent ID reused for different operation",
            )
        };
    }
    if let Some(key) = operation_key(intent) {
        if let Some(existing) = state.operation_keys.get(&key) {
            let record = state
                .intents
                .get(existing)
                .ok_or("idempotency_corruption")?;
            return if same_operation(&record.intent, intent) {
                Ok(json!({"duplicate":true,"record":record}))
            } else {
                fail(
                    "idempotency_collision",
                    "operation key reused for different action arguments/resources",
                )
            };
        }
    }
    if state.intents.len() >= config.max_intents {
        return fail(
            "intent_limit",
            "intent ledger full; archive rather than forgetting duplicate identities",
        );
    }
    expire(state, now);
    check_authority(state, intent)?;
    for record in state.intents.values().filter(|r| !r.status.terminal()) {
        if conflicts(&record.intent, intent) {
            return fail("lease_conflict", &record.intent.intent_id);
        }
    }
    let record = IntentRecord {
        intent: intent.clone(),
        status: Status::Accepted,
        accepted_ms: now,
        expires_ms: now.checked_add(intent.ttl_ms).ok_or("deadline_overflow")?,
        started_ms: None,
        finished_ms: None,
        cancellation_requested: false,
        outcome: None,
        invalidation_reason: None,
    };
    if let Some(key) = operation_key(intent) {
        state.operation_keys.insert(key, intent.intent_id.clone());
    }
    state
        .intents
        .insert(intent.intent_id.clone(), record.clone());
    Ok(json!({"duplicate":false,"record":record}))
}

fn begin(state: &mut State, intent_id: &str, now: u64) -> Result<Value> {
    let record = state
        .intents
        .get(intent_id)
        .ok_or_else(|| format!("unknown_intent: {intent_id}"))?
        .clone();
    if record.status != Status::Accepted {
        return fail("not_dispatchable", format!("{:?}", record.status));
    }
    if record.expires_ms <= now {
        return fail("expired", "intent deadline passed before dispatch");
    }
    if record.cancellation_requested {
        return fail("cancelled", "cancellation already requested");
    }
    let spec = check_authority(state, &record.intent)?;
    for other in state
        .intents
        .values()
        .filter(|r| !r.status.terminal() && r.intent.intent_id != intent_id)
    {
        if conflicts(&other.intent, &record.intent) {
            return fail("lease_conflict", &other.intent.intent_id);
        }
    }
    let target = state
        .intents
        .get_mut(intent_id)
        .ok_or("intent_disappeared")?;
    target.status = Status::Running;
    target.started_ms = Some(now);
    Ok(
        json!({"admitted":true,"record":target,"action_spec":spec,"arguments":target.intent.arguments}),
    )
}

fn finish(
    state: &mut State,
    intent_id: &str,
    outcome: &Outcome,
    now: u64,
    reconcile: bool,
) -> Result<Value> {
    let record = state
        .intents
        .get(intent_id)
        .ok_or_else(|| format!("unknown_intent: {intent_id}"))?
        .clone();
    if record.status.terminal() {
        return if record.outcome.as_ref() == Some(outcome) {
            value(&record)
        } else {
            fail("terminal_intent", "cannot replace a terminal receipt")
        };
    }
    let valid = if reconcile {
        matches!(
            record.status,
            Status::EffectUnknown | Status::CancellationRequested
        )
    } else {
        matches!(
            record.status,
            Status::Running | Status::CancellationRequested
        )
    };
    if !valid {
        return fail(
            "outcome_transition",
            "finish requires running; reconcile requires unknown/cancel-requested",
        );
    }
    if !outcome.evidence.is_object() || outcome.evidence.as_object().is_some_and(|m| m.is_empty()) {
        return fail(
            "missing_evidence",
            "outcome must carry a nonempty verifier/receipt evidence object",
        );
    }
    if outcome
        .resource_fingerprints
        .keys()
        .any(|key| !outcome.changed_resources.contains(key))
    {
        return fail(
            "effect_mismatch",
            "fingerprints must correspond to changed resources",
        );
    }
    for resource in &outcome.changed_resources {
        if !record
            .intent
            .resources
            .iter()
            .any(|r| &r.resource_id == resource && r.access == Access::Write)
        {
            return fail("effect_scope", resource);
        }
        let current = state
            .resources
            .get_mut(resource)
            .ok_or("unknown_changed_resource")?;
        current.revision = current.revision.checked_add(1).ok_or("revision_overflow")?;
        if let Some(fp) = outcome.resource_fingerprints.get(resource) {
            if fp.len() > 4096 {
                return fail("fingerprint_limit", "fingerprint too long");
            }
            current.fingerprint = fp.clone();
        }
        current.observed_ms = now;
    }
    let target = state
        .intents
        .get_mut(intent_id)
        .ok_or("intent_disappeared")?;
    target.status = match outcome.status {
        OutcomeStatus::CompletedVerified => Status::CompletedVerified,
        OutcomeStatus::FailedVerified => Status::FailedVerified,
        OutcomeStatus::CancelledVerified => Status::Cancelled,
        OutcomeStatus::EffectUnknown => Status::EffectUnknown,
    };
    if target.status.terminal() {
        target.finished_ms = Some(now);
    }
    target.outcome = Some(outcome.clone());
    value(target)
}

fn cancel(state: &mut State, intent_id: &str, now: u64) -> Result<Value> {
    let record = state
        .intents
        .get_mut(intent_id)
        .ok_or_else(|| format!("unknown_intent: {intent_id}"))?;
    if !record.status.terminal() {
        record.cancellation_requested = true;
        match record.status {
            Status::Accepted => {
                record.status = Status::Cancelled;
                record.finished_ms = Some(now);
            }
            Status::Running => record.status = Status::CancellationRequested,
            _ => {}
        }
    }
    value(record)
}

fn revoke(state: &mut State, capability: &str, now: u64) -> Result<Value> {
    id(capability)?;
    let removed = state.capabilities.remove(capability);
    let affected: Vec<String> = state
        .intents
        .iter()
        .filter(|(_, record)| {
            !record.status.terminal()
                && state
                    .actions
                    .get(&record.intent.action_id)
                    .is_some_and(|spec| spec.required_capabilities.contains(capability))
        })
        .map(|(key, _)| key.clone())
        .collect();
    for key in &affected {
        cancel(state, key, now)?;
    }
    Ok(json!({"revoked":removed,"capability":capability,"affected":affected}))
}

fn expire(state: &mut State, now: u64) -> Vec<String> {
    let mut expired = vec![];
    for (key, record) in &mut state.intents {
        if record.status == Status::Accepted && record.expires_ms <= now {
            record.status = Status::Expired;
            record.finished_ms = Some(now);
            expired.push(key.clone());
        }
    }
    expired
}
