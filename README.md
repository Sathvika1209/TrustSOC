# TrustSOC

TrustSOC is a prototype security-operations workflow for investigating
network-flow alerts with MITRE ATT&CK reference material. It is intentionally
conservative: retrieval produces investigation candidates, not an attack
finding, risk score, confidence score, priority, or malicious/benign verdict.

## Architecture

The current path is:

```text
CICIDS-style processed CSV
  -> normalized network-flow alert
  -> SentenceTransformers query embedding
  -> FAISS MITRE ATT&CK retrieval
  -> ATT&CK detection-strategy enrichment
  -> evidence-gap report
  -> conservative decision record
```

`scripts/demo_threat_intel.py` contains the retrieval and evidence-validation
implementation. `agents/threat_intel.py` wraps it for reuse and loads the
corpus, embedding model, and index once. `decision/decision_layer.py`
aggregates agent reports and renders a structured or plain-text result.
`scripts/run_trustsoc.py` is the end-to-end CLI.

The NVD/CVE pilot corpus at
`data/knowledge_base/nvd_cve/nvd_cve_documents_pilot.json` is reference data
only. It is **not connected to the runtime retrieval pipeline, agent, or
decision layer** in this phase.

## Current capabilities and limitations

- Retrieves top-k MITRE ATT&CK documents using `all-MiniLM-L6-v2` and
  FAISS `IndexFlatL2`.
- Adds ATT&CK detection strategies, analytics, and example log sources as
  investigation guidance.
- Reports available network-flow telemetry and missing endpoint, identity,
  asset/vulnerability, and security-event context.
- Does not inspect endpoint logs, identity logs, assets, vulnerabilities, or
  raw packets.
- Does not establish that a technique occurred.
- Retrieval distance is not a probability, confidence, risk, or priority.
- The benchmark labels are provisional manual judgments and evaluate
  retrieval only, not detection or technique applicability.

## Setup

Use CPython **3.12** and run commands from the repository root. CPython 3.12.3
was the only version tested for this phase; Python 3.10-3.13 are not claimed
as verified compatibility targets.

Install the direct dependencies and the tested ML constraints:

```bash
python -m pip install -r requirements.txt -c constraints-py312.txt
```

`requirements.txt` contains the direct runtime/notebook dependencies: NumPy,
pandas, `faiss-cpu`, Sentence Transformers, and requests. The test scripts use
Python's standard-library `unittest`; no separate test package is required.
`constraints-py312.txt` records the ML transitive versions tested with those
direct dependencies: PyTorch, Transformers, Tokenizers, and Safetensors.

For a CPU-only installation, install the CPU PyTorch wheel before applying the
project requirements. The command below uses the official PyTorch CPU index;
the remaining packages come from the normal package index:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1
.venv/bin/python -m pip install \
  -r requirements.txt -c constraints-py312.txt
```

The CPU-only bootstrap was not executable in the restricted review environment
because external package access is not guaranteed. The exact non-model package
resolution was tested previously in an isolated Python 3.12 environment, but
the CPU-wheel variant and Python versions other than 3.12 remain unverified.

The embedding model may be downloaded from the Hugging Face model hub on its
first use; network access and a writable model cache are required unless the
model is already cached.

## Data requirements

The committed MITRE artifacts are:

- `data/knowledge_base/mitre_attack/mitre_attack_documents.json`
- `data/knowledge_base/mitre_attack/detection_evidence.json`

The runtime also requires a processed CICIDS-style CSV at:

```text
data/processed/Friday-WorkingHours-Morning.pcap_ISCX_recovered.csv
```

The CSV must contain the flow columns read by `create_soc_alert()` in
`scripts/demo_threat_intel.py`. Its `Label` column is deliberately excluded
from alerts. Raw CICIDS data and processed CSVs are not committed; place them
locally or pass an alternative processed CSV with `--file` to
`scripts/run_trustsoc.py`.

The notebook documents the preprocessing flow and discovers the repository
root when launched either from the repository root or from `notebooks/`. Set
`TRUSTSOC_RAW_DATA_DIR`, `TRUSTSOC_PROCESSED_DATA_DIR`, or
`TRUSTSOC_KNOWLEDGE_BASE_DIR` to use external local data directories. The
notebook does not generate runtime source code: the checked-in
`scripts/demo_threat_intel.py` is the source of truth. Downloaded source data
and generated outputs are not committed.

## FAISS index and metadata workflow

The active artifact is selected by this manifest:

```text
data/knowledge_base/mitre_attack/mitre_attack.current.json
```

The manifest points to immutable generation-specific files such as:

```text
data/knowledge_base/mitre_attack/mitre_attack.<generation>.faiss
data/knowledge_base/mitre_attack/mitre_attack.<generation>.metadata.json
```

All of these files are generated and ignored by Git. A new generation is
written and validated before the manifest is atomically replaced, so a failed
build or interruption during publication leaves the previously active
generation usable. The manifest records both the active generation and its
explicitly designated previous generation. If the active generation is
missing, corrupt, stale, or incompatible, loaders validate only that
designated previous generation; they never select an arbitrary stale file.
Malformed or unreadable manifests are reported as errors rather than silently
recovered. A missing manifest is reported distinctly and permits a fresh
build. Unreferenced stale generations are retained deliberately; cleanup is
not automatic and must never remove the active or designated fallback
generation.

Generation metadata records schema version 1, the ordered corpus SHA-256
fingerprint, model name, vector dimension, document count, index type, and L2
metric. Evaluation scripts use the same manifest/generation workflow and
validate these fields before searching. The fingerprint retains the original
builder's byte separator for compatibility. Metadata without schema version 1
is rebuilt into the current format. The tests mock publication failures and
validate filesystem replacement behavior; they do not simulate actual power
loss. Run the demo once to create the artifact:

```bash
python scripts/demo_threat_intel.py --sample 1 --top-k 5
```

Do not commit the generated index, metadata, model weights, or datasets.

## Runtime commands

Human-readable decision output:

```bash
python scripts/run_trustsoc.py --sample 1 --top-k 5
```

Structured decision JSON:

```bash
python scripts/run_trustsoc.py --sample 1 --top-k 5 --json
```

Use another local processed CSV:

```bash
python scripts/run_trustsoc.py --file path/to/processed.csv --sample 1
```

The standalone demo prints the detailed threat-intelligence report:

```bash
python scripts/demo_threat_intel.py --sample 1 --top-k 5
```

## Evaluation

After building the canonical FAISS manifest and generation:

```bash
python scripts/evaluate_retrieval.py --top-k 10
python scripts/evaluate_labeled_retrieval.py
python scripts/evaluate_query_expansion.py
python scripts/experiment_query_expansion.py
```

The query-expansion scripts are experiments only and do not change production
retrieval behavior. `evaluation/retrieval_benchmark.json` contains 15
provisional cases.

## Tests and checks

The repository currently uses executable test scripts rather than a configured
test runner:

```bash
python scripts/test_detection_evidence.py
python scripts/test_threat_intel.py
python scripts/test_decision.py
python -m py_compile agents/*.py decision/*.py scripts/*.py
```

The integration portions of the threat-intelligence and decision tests need
the dependencies, embedding model, and processed CSV. They may take time on
the first run because the model and FAISS index are initialized.

## Reproducibility gaps

This phase does not provide a service API. Dependency versions are pinned for
the tested direct set and key ML transitive packages, but a complete
platform-independent lockfile is not provided because PyTorch CPU/CUDA wheels
are platform-specific. The notebook still requires external
downloads for raw datasets and ATT&CK source data. The source ATT&CK STIX file
needed by `scripts/build_detection_evidence.py` is not committed; obtain it
locally before regenerating detection evidence. Reproducible environment
locking and runtime integration of NVD/CVE data remain future work.
