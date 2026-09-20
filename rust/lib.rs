//! Native authority/state reducer. No module executes filesystem, process, or network effects.
mod contracts;
mod core;
mod policy;
#[cfg(test)]
mod tests;

use contracts::*;
use core::{Core, Journal};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use std::sync::Mutex;
use std::time::Instant;

pyo3::create_exception!(_native, BrokerError, PyRuntimeError);
fn error(message: String) -> PyErr {
    BrokerError::new_err(message)
}
fn decode<T: DeserializeOwned>(text: &str, limit: usize) -> PyResult<T> {
    if text.len() > limit {
        return Err(error("payload_limit: JSON exceeds boundary size".into()));
    }
    serde_json::from_str(text).map_err(|e| error(format!("invalid_json: {e}")))
}

#[pyclass(frozen, module = "reflexmesh._native")]
pub struct Broker {
    inner: Mutex<Core>,
    origin: Instant,
    offset_ms: u64,
    max_payload: usize,
}
impl Broker {
    fn from_core(core: Core) -> Self {
        let offset_ms = core.tick_ms;
        let max_payload = core.config.max_event_bytes;
        Self {
            inner: Mutex::new(core),
            origin: Instant::now(),
            offset_ms,
            max_payload,
        }
    }
    fn tick(&self) -> u64 {
        self.offset_ms
            .saturating_add(self.origin.elapsed().as_millis().min(u64::MAX as u128) as u64)
    }
    fn command(&self, py: Python<'_>, command: Command) -> PyResult<String> {
        py.detach(|| {
            let mut core = self
                .inner
                .lock()
                .map_err(|_| "broker_poisoned".to_string())?;
            let result = core.transact(command, self.tick())?;
            serde_json::to_string(&result).map_err(|e| e.to_string())
        })
        .map_err(error)
    }
}

#[pymethods]
impl Broker {
    #[new]
    #[pyo3(signature = (config_json="{}"))]
    fn new(config_json: &str) -> PyResult<Self> {
        let config: Config = decode(config_json, 1048576)?;
        Ok(Self::from_core(Core::new(config).map_err(error)?))
    }

    fn register_action(&self, py: Python<'_>, spec_json: &str) -> PyResult<String> {
        self.command(
            py,
            Command::RegisterAction {
                spec: decode(spec_json, self.max_payload)?,
            },
        )
    }
    fn observe(&self, py: Python<'_>, event_json: &str) -> PyResult<String> {
        self.command(
            py,
            Command::Observe {
                event: decode(event_json, self.max_payload)?,
            },
        )
    }
    fn submit(&self, py: Python<'_>, intent_json: &str) -> PyResult<String> {
        self.command(
            py,
            Command::Submit {
                intent: decode(intent_json, self.max_payload)?,
            },
        )
    }
    fn begin(&self, py: Python<'_>, intent_id: String) -> PyResult<String> {
        self.command(py, Command::Begin { intent_id })
    }
    fn finish(&self, py: Python<'_>, intent_id: String, outcome_json: &str) -> PyResult<String> {
        self.command(
            py,
            Command::Finish {
                intent_id,
                outcome: decode(outcome_json, self.max_payload)?,
            },
        )
    }
    fn reconcile(&self, py: Python<'_>, intent_id: String, outcome_json: &str) -> PyResult<String> {
        self.command(
            py,
            Command::Reconcile {
                intent_id,
                outcome: decode(outcome_json, self.max_payload)?,
            },
        )
    }
    fn cancel(&self, py: Python<'_>, intent_id: String) -> PyResult<String> {
        self.command(py, Command::Cancel { intent_id })
    }
    fn revoke(&self, py: Python<'_>, capability: String) -> PyResult<String> {
        self.command(py, Command::Revoke { capability })
    }
    fn expire(&self, py: Python<'_>) -> PyResult<String> {
        self.command(py, Command::Expire)
    }
    #[pyo3(signature = (event_json, control=false))]
    fn enqueue(&self, py: Python<'_>, event_json: &str, control: bool) -> PyResult<String> {
        self.command(
            py,
            Command::Enqueue {
                event: decode(event_json, self.max_payload)?,
                control,
            },
        )
    }
    #[pyo3(signature = (limit=1))]
    fn drain(&self, py: Python<'_>, limit: usize) -> PyResult<String> {
        self.command(py, Command::Drain { limit })
    }
    fn now_ms(&self) -> u64 {
        self.tick()
    }
    fn snapshot(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            let core = self
                .inner
                .lock()
                .map_err(|_| "broker_poisoned".to_string())?;
            serde_json::to_string(&core.snapshot()).map_err(|e| e.to_string())
        })
        .map_err(error)
    }
    fn journal(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            let core = self
                .inner
                .lock()
                .map_err(|_| "broker_poisoned".to_string())?;
            serde_json::to_string(&core.journal).map_err(|e| e.to_string())
        })
        .map_err(error)
    }
    fn checkpoint(&self, py: Python<'_>, acknowledged_tip: &str) -> PyResult<String> {
        py.detach(|| {
            let mut core = self
                .inner
                .lock()
                .map_err(|_| "broker_poisoned".to_string())?;
            serde_json::to_string(&core.checkpoint(acknowledged_tip)?).map_err(|e| e.to_string())
        })
        .map_err(error)
    }
    /// Deterministic, observational replay only. This returns state and cannot dispatch effects.
    #[staticmethod]
    fn replay(py: Python<'_>, journal_json: &str) -> PyResult<String> {
        let journal: Journal = decode(journal_json, 268435456)?;
        py.detach(|| {
            let core = Core::replay(journal)?;
            serde_json::to_string(&core.snapshot()).map_err(|e| e.to_string())
        })
        .map_err(error)
    }
    /// Recover authority after a restart. Running effects become unknown; pending dispatches invalidate.
    #[staticmethod]
    fn from_journal(py: Python<'_>, journal_json: &str) -> PyResult<Self> {
        let journal: Journal = decode(journal_json, 268435456)?;
        let core = py
            .detach(|| {
                let mut core = Core::replay(journal)?;
                core.transact(Command::Recover, core.tick_ms.saturating_add(1))?;
                Ok::<_, String>(core)
            })
            .map_err(error)?;
        Ok(Self::from_core(core))
    }
}

#[pyfunction]
fn score_linear(py: Python<'_>, export_json: &str, features_json: &str) -> PyResult<String> {
    py.detach(|| policy::score_linear(export_json, features_json))
        .map_err(error)
}

#[pyfunction]
fn score_trees(py: Python<'_>, export_json: &str, features_json: &str) -> PyResult<String> {
    py.detach(|| policy::score_trees(export_json, features_json))
        .map_err(error)
}

#[pyclass(frozen, module = "reflexmesh._native")]
pub struct LinearScorer {
    model: policy::LinearExport,
}

#[pymethods]
impl LinearScorer {
    #[new]
    fn new(py: Python<'_>, export_json: &str) -> PyResult<Self> {
        let model = py
            .detach(|| policy::load_linear(export_json))
            .map_err(error)?;
        Ok(Self { model })
    }
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model.feature_names.clone()
    }
    fn score(&self, py: Python<'_>, features_json: &str) -> PyResult<String> {
        py.detach(|| policy::linear_json(&self.model, features_json))
            .map_err(error)
    }
    fn score_f32(
        &self,
        py: Python<'_>,
        features_le_bytes: &[u8],
        row_count: usize,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            let rows =
                policy::packed_rows(features_le_bytes, row_count, self.model.feature_names.len())?;
            let rows: Vec<Vec<f64>> = rows
                .into_iter()
                .map(|row| row.into_iter().map(f64::from).collect())
                .collect();
            policy::linear_rows(&self.model, &rows)
        })
        .map_err(error)
    }
}

#[pyclass(frozen, module = "reflexmesh._native")]
pub struct TreeScorer {
    model: policy::TreeExport,
}

#[pymethods]
impl TreeScorer {
    #[new]
    fn new(py: Python<'_>, export_json: &str) -> PyResult<Self> {
        let model = py
            .detach(|| policy::load_trees(export_json))
            .map_err(error)?;
        Ok(Self { model })
    }
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model.feature_names.clone()
    }
    fn score(&self, py: Python<'_>, features_json: &str) -> PyResult<String> {
        py.detach(|| policy::trees_json(&self.model, features_json))
            .map_err(error)
    }
    fn score_f32(
        &self,
        py: Python<'_>,
        features_le_bytes: &[u8],
        row_count: usize,
    ) -> PyResult<Vec<f64>> {
        py.detach(|| {
            let rows =
                policy::packed_rows(features_le_bytes, row_count, self.model.feature_names.len())?;
            policy::tree_rows(&self.model, &rows)
        })
        .map_err(error)
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Broker>()?;
    module.add_class::<LinearScorer>()?;
    module.add_class::<TreeScorer>()?;
    module.add("BrokerError", module.py().get_type::<BrokerError>())?;
    module.add_function(wrap_pyfunction!(score_linear, module)?)?;
    module.add_function(wrap_pyfunction!(score_trees, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__engine__", "rust-pyo3")?;
    Ok(())
}
