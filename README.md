# automl-reusables

Shared Python library used by every AutoML microservice.

Contains:

- `src.contracts` — Pydantic schemas (`MetaFeatures`, `RunResult`, `GeneratedPipeline`, `ErrorCategory`, `TaskType`)
- `src.conditions` — prompt conditions (B0 naive, B1 schema, B2 meta-feature-guided)
- `src.meta_features` — dataset meta-feature extraction (simple, distributional, information, landmarking)
- `src.execution` — subprocess-isolated pipeline execution and error taxonomy
- `src.experiments` — dataset loading (OpenML + custom CSV), LLM orchestration, analysis functions

## Install (editable, from a sibling project)

```
pip install -e ../automl-reusables
```

After install, the modules are importable as:

```python
from src.contracts import MetaFeatures, RunResult
from src.meta_features.extractor import extract as extract_meta
from src.execution.runner import execute_pipeline
```

The importable module name is `src` for backward-compatibility with existing code.
The installable package name is `automl-reusables`.

## Consumers

- `major-project-AutoML` — research CLI (`dry_run.py`, `run_sweep.py`, `tests/`)
- `automl-data-service` — dataset upload / registry (port 8001)
- `automl-metafeatures-service` — meta-feature extraction API (port 8002)
- `automl-generation-service` — async LLM generation via RQ (port 8003)
- `automl-analysis-service` — analysis / stats API (port 8004)
- `automl-gateway` — API gateway for the frontend (port 8000)

## License

Dual-licensed under either of

- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution you
intentionally submit for inclusion in this work shall be dual-licensed as
above, without any additional terms or conditions.

## Citing

Part of the VESIT AutoML project. For citation metadata and (once released) a
DOI, see the [`automl-infra`](https://github.com/Major-Proj-AutoML/automl-infra)
repository's `CITATION.cff`.
