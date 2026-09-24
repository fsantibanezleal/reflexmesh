use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Calibration {
    kind: String,
    coef: f64,
    intercept: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearExport {
    schema_version: u32,
    pub feature_names: Vec<String>,
    coef: Vec<f64>,
    intercept: f64,
    calibration: Calibration,
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}
pub fn load_linear(export: &str) -> Result<LinearExport, String> {
    if export.len() > 1048576 {
        return Err("policy_payload_limit".into());
    }
    let model: LinearExport =
        serde_json::from_str(export).map_err(|e| format!("invalid_linear_export: {e}"))?;
    if model.schema_version != 1
        || model.feature_names.len() != model.coef.len()
        || model.coef.is_empty()
        || model.coef.len() > 4096
        || model.calibration.kind != "platt"
        || !model.intercept.is_finite()
        || !model.calibration.coef.is_finite()
        || !model.calibration.intercept.is_finite()
        || model.coef.iter().any(|v| !v.is_finite())
    {
        return Err(
            "invalid_linear_export: unsupported schema or nonfinite/dimension bounds".into(),
        );
    }
    Ok(model)
}

pub fn linear_rows(model: &LinearExport, rows: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    if rows.len() > 4096 {
        return Err("policy_row_limit".into());
    }
    let mut scores = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != model.coef.len() || row.iter().any(|v| !v.is_finite()) {
            return Err("invalid_features: wrong dimension or nonfinite number".into());
        }
        let logit = model
            .coef
            .iter()
            .zip(row.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + model.intercept;
        let calibrated = model.calibration.coef * logit + model.calibration.intercept;
        if !logit.is_finite() || !calibrated.is_finite() {
            return Err("numerical_overflow".into());
        }
        scores.push(sigmoid(calibrated));
    }
    Ok(scores)
}

pub fn linear_json(model: &LinearExport, rows: &str) -> Result<String, String> {
    if rows.len() > 8388608 {
        return Err("policy_payload_limit".into());
    }
    let rows: Vec<Vec<f64>> =
        serde_json::from_str(rows).map_err(|e| format!("invalid_features: {e}"))?;
    serde_json::to_string(&linear_rows(model, &rows)?).map_err(|e| e.to_string())
}

pub fn score_linear(export: &str, rows: &str) -> Result<String, String> {
    linear_json(&load_linear(export)?, rows)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Leaf {
    node_id: usize,
    leaf: f32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Split {
    node_id: usize,
    feature_index: usize,
    threshold: f32,
    left: usize,
    right: usize,
    missing: usize,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Node {
    Leaf(Leaf),
    Split(Split),
}
impl Node {
    fn id(&self) -> usize {
        match self {
            Self::Leaf(node) => node.node_id,
            Self::Split(node) => node.node_id,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tree {
    nodes: Vec<Node>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreeExport {
    schema_version: u32,
    pub feature_names: Vec<String>,
    base_margin: f32,
    trees: Vec<Tree>,
    calibration: Calibration,
}

/// Canonical XGBoost-style binary trees. Float32 comparison/accumulation mirrors
/// the CPU predictor; probability calibration is applied in float64.
pub fn load_trees(export: &str) -> Result<TreeExport, String> {
    if export.len() > 67108864 {
        return Err("policy_payload_limit".into());
    }
    let model: TreeExport =
        serde_json::from_str(export).map_err(|e| format!("invalid_tree_export: {e}"))?;
    if model.schema_version != 1
        || model.feature_names.is_empty()
        || model.feature_names.len() > 4096
        || model.trees.is_empty()
        || model.trees.len() > 4096
        || !model.base_margin.is_finite()
        || model.calibration.kind != "platt"
        || !model.calibration.coef.is_finite()
        || !model.calibration.intercept.is_finite()
    {
        return Err("invalid_tree_export: unsupported schema or bounds".into());
    }
    let mut total_nodes = 0usize;
    for tree in &model.trees {
        total_nodes = total_nodes.saturating_add(tree.nodes.len());
        if tree.nodes.is_empty() || total_nodes > 1_000_000 {
            return Err("invalid_tree_export: node bounds".into());
        }
        let mut parents = vec![0usize; tree.nodes.len()];
        for (index, node) in tree.nodes.iter().enumerate() {
            if node.id() != index {
                return Err("invalid_tree_export: IDs must be contiguous".into());
            }
            match node {
                Node::Leaf(leaf) if !leaf.leaf.is_finite() => {
                    return Err("invalid_tree_export: nonfinite leaf".into())
                }
                Node::Leaf(_) => {}
                Node::Split(split) => {
                    if split.feature_index >= model.feature_names.len()
                        || !split.threshold.is_finite()
                        || split.left <= index
                        || split.right <= index
                        || split.left == split.right
                        || split.left >= tree.nodes.len()
                        || split.right >= tree.nodes.len()
                        || (split.missing != split.left && split.missing != split.right)
                    {
                        return Err("invalid_tree_export: invalid split or cyclic graph".into());
                    }
                    parents[split.left] += 1;
                    parents[split.right] += 1;
                }
            }
        }
        if parents[0] != 0 || parents.iter().skip(1).any(|count| *count != 1) {
            return Err("invalid_tree_export: graph must be one rooted tree".into());
        }
    }
    Ok(model)
}

pub fn tree_rows(model: &TreeExport, rows: &[Vec<f32>]) -> Result<Vec<f64>, String> {
    if rows.len() > 4096 {
        return Err("policy_row_limit".into());
    }
    let mut scores = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != model.feature_names.len() || row.iter().any(|v| !v.is_finite()) {
            return Err("invalid_features: wrong dimension or nonfinite number".into());
        }
        let mut margin = model.base_margin;
        for tree in &model.trees {
            let mut current = 0;
            loop {
                match &tree.nodes[current] {
                    Node::Leaf(leaf) => {
                        margin += leaf.leaf;
                        break;
                    }
                    Node::Split(split) => {
                        current = if row[split.feature_index] < split.threshold {
                            split.left
                        } else {
                            split.right
                        };
                    }
                }
            }
        }
        let calibrated = model.calibration.coef * f64::from(margin) + model.calibration.intercept;
        if !margin.is_finite() || !calibrated.is_finite() {
            return Err("numerical_overflow".into());
        }
        scores.push(sigmoid(calibrated));
    }
    Ok(scores)
}

pub fn trees_json(model: &TreeExport, rows: &str) -> Result<String, String> {
    if rows.len() > 8388608 {
        return Err("policy_payload_limit".into());
    }
    let rows: Vec<Vec<f32>> =
        serde_json::from_str(rows).map_err(|e| format!("invalid_features: {e}"))?;
    serde_json::to_string(&tree_rows(model, &rows)?).map_err(|e| e.to_string())
}

pub fn score_trees(export: &str, rows: &str) -> Result<String, String> {
    trees_json(&load_trees(export)?, rows)
}

/// Fixed-endian packed float32 boundary: no alignment/aliasing assumptions.
pub fn packed_rows(bytes: &[u8], count: usize, width: usize) -> Result<Vec<Vec<f32>>, String> {
    if count > 4096 || width == 0 || width > 4096 || bytes.len() > 8388608 {
        return Err("policy_payload_limit".into());
    }
    let expected = count.checked_mul(width).and_then(|v| v.checked_mul(4));
    if expected != Some(bytes.len()) {
        return Err("invalid_features: packed byte length does not match dimensions".into());
    }
    let mut rows = Vec::with_capacity(count);
    for row in bytes.chunks_exact(width * 4) {
        let values: Vec<f32> = row
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("invalid_features: nonfinite packed value".into());
        }
        rows.push(values);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    const EXPORT: &str = r#"{"schema_version":1,"feature_names":["x","y"],"coef":[2,-1],"intercept":0.5,"calibration":{"kind":"platt","coef":1.2,"intercept":-0.3}}"#;
    #[test]
    fn calibrated_export_and_invalid_dimensions() {
        let scores: Vec<f64> =
            serde_json::from_str(&score_linear(EXPORT, "[[0,0],[1,2],[-1000,1000]]").unwrap())
                .unwrap();
        assert!((scores[0] - sigmoid(0.3)).abs() < 1e-14);
        assert_eq!(scores[0], scores[1]);
        assert_eq!(scores[2], 0.0);
        assert!(score_linear(EXPORT, "[[0]]").is_err());
        assert!(score_linear(EXPORT, "[[1e308,-1e308]]").is_err());
    }

    #[test]
    fn trees_use_strict_split_boundaries_and_validate_graphs() {
        let export = r#"{"schema_version":1,"feature_names":["x"],"base_margin":0,
            "trees":[{"nodes":[{"node_id":0,"feature_index":0,"threshold":1,
            "left":1,"right":2,"missing":1},{"node_id":1,"leaf":-2},{"node_id":2,"leaf":2}]}],
            "calibration":{"kind":"platt","coef":1,"intercept":0}}"#;
        let scores: Vec<f64> =
            serde_json::from_str(&score_trees(export, "[[0.9999],[1],[1.0001]]").unwrap()).unwrap();
        assert!((scores[0] - sigmoid(-2.0)).abs() < 1e-14);
        assert_eq!(scores[1], sigmoid(2.0));
        assert_eq!(scores[1], scores[2]);
        let cycle = export.replace("\"left\":1", "\"left\":0");
        assert!(score_trees(&cycle, "[[0]]").unwrap_err().contains("cyclic"));
        let disconnected = export.replace("\"node_id\":1", "\"node_id\":99");
        assert!(score_trees(&disconnected, "[[0]]").is_err());
        assert!(score_trees(export, "[[1e308]]").is_err());
    }

    #[test]
    fn cached_models_and_packed_boundary_are_validated() {
        let model = load_linear(EXPORT).unwrap();
        let bytes: Vec<u8> = [0.0f32, 0.0, 1.0, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let packed = packed_rows(&bytes, 2, 2).unwrap();
        let rows = packed
            .into_iter()
            .map(|row| row.into_iter().map(f64::from).collect())
            .collect::<Vec<Vec<f64>>>();
        assert_eq!(linear_rows(&model, &rows).unwrap()[0], sigmoid(0.3));
        assert!(packed_rows(&bytes, 3, 2).is_err());
        assert!(packed_rows(&bytes, 2, 0).is_err());
        assert!(packed_rows(&f32::NAN.to_le_bytes(), 1, 1).is_err());
        assert!(packed_rows(&f32::INFINITY.to_le_bytes(), 1, 1).is_err());
        assert!(packed_rows(&[], 4097, 1).is_err());
    }
}
