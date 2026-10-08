# Authenticated semantic execution

`GET /api/config` discovers the core and semantic_transfer suites. `POST /api/run` accepts an optional `suite` field; omission retains the core contract. For semantic_transfer, select D01–D08, nominal or boundary, an integer seed from 0 through 999999 and one of M01–M12. Only `max_steps` (1–64) is accepted in parameters.

The service constructs SemanticSpec and executes SemanticEnvironment through the normal episode runner. Real JSON files, registered native capabilities, revision checks, independent verification, cancellation, retained-run limits and bearer/same-origin authorization remain active. Numeric tool features do not expose the correct operation; candidate descriptions and the observed goal carry semantic information. A success requires all three independently checked output files. A receipt alone is insufficient.

Fresh episodes retain their suite lane and identify `execution_origin: fresh-local` and `suite` in provenance. They do not rewrite or append to canonical benchmark artifacts. Stream results from `GET /api/events?run_id=...`; request cancellation with `POST /api/cancel/{run_id}`. Missing planner prerequisites, unsupported fields and invalid suite/family/variant combinations fail explicitly.

Example request, with a session bearer token supplied by the operator:

```json
{"suite":"semantic_transfer","case_id":"D01","method_id":"M01","variant":"nominal","seed":82000}
```

M01 may abstain because the semantic capabilities do not match its state rules. That outcome is retained as a failure. The API exposes the existing research methods and does not claim a new trained controller or improved benchmark result.
