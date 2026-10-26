"""Shared paths, validation, and lifecycle for MITRE retrieval artifacts."""

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path


MODEL_NAME = "all-MiniLM-L6-v2"
METADATA_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
EXPECTED_INDEX_TYPE = "IndexFlatL2"
EXPECTED_METRIC_TYPE = "METRIC_L2"
PROJECT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_DIR / "data" / "knowledge_base" / "mitre_attack"
DOCS_PATH = KB_DIR / "mitre_attack_documents.json"
MANIFEST_PATH = KB_DIR / "mitre_attack.current.json"


class ArtifactError(RuntimeError):
    """An expected missing, malformed, stale, or corrupt artifact error."""


def corpus_fingerprint(documents):
    """Hash the ordered document content used by the original index builder."""
    digest = hashlib.sha256()
    for document in documents:
        canonical = json.dumps(
            document,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\\n")
    return digest.hexdigest()


def _index_metric_name(faiss_module, index):
    metric_l2 = getattr(faiss_module, "METRIC_L2", 1)
    if getattr(index, "metric_type", None) != metric_l2:
        return None
    return EXPECTED_METRIC_TYPE


def validate_index(
    index,
    documents,
    metadata,
    faiss_module,
    model_name=MODEL_NAME,
):
    """Validate a FAISS index and its metadata against the corpus."""
    if not isinstance(metadata, dict):
        raise ArtifactError("FAISS metadata must be a JSON object.")
    if metadata.get("schema_version") != METADATA_SCHEMA_VERSION:
        raise ArtifactError("Unsupported or missing FAISS metadata schema.")
    if metadata.get("index_type") != EXPECTED_INDEX_TYPE:
        raise ArtifactError("FAISS metadata index type is not supported.")
    if metadata.get("metric_type") != EXPECTED_METRIC_TYPE:
        raise ArtifactError("FAISS metadata metric type is not supported.")
    if index.__class__.__name__ != EXPECTED_INDEX_TYPE:
        raise ArtifactError("FAISS index type is not IndexFlatL2.")
    if _index_metric_name(faiss_module, index) != EXPECTED_METRIC_TYPE:
        raise ArtifactError("FAISS index metric is not L2.")
    if metadata.get("corpus_fingerprint") != corpus_fingerprint(documents):
        raise ArtifactError("FAISS metadata fingerprint does not match the corpus.")
    if metadata.get("model_name") != model_name:
        raise ArtifactError("Embedding model does not match FAISS metadata.")
    if metadata.get("document_count") != len(documents):
        raise ArtifactError("Document count does not match FAISS metadata.")
    if metadata.get("vector_dimension") != index.d:
        raise ArtifactError("Vector dimension does not match FAISS metadata.")
    if index.ntotal != len(documents):
        raise ArtifactError("FAISS index size differs from document count.")
    return index


def _read_json(path, description):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"{description} could not be read: {path}") from error
    return value


def _read_index(faiss_module, path):
    try:
        return faiss_module.read_index(str(path))
    except (OSError, ValueError, RuntimeError) as error:
        raise ArtifactError(f"FAISS index could not be read: {path}") from error


def _generation_paths(generation):
    valid_uuid = False
    if isinstance(generation, str) and generation:
        try:
            valid_uuid = uuid.UUID(generation).hex == generation
        except ValueError:
            pass
    if not valid_uuid:
        raise ArtifactError(
            "Invalid FAISS generation identifier; expected a 32-character "
            "lowercase hexadecimal UUID."
        )
    artifact_dir = KB_DIR.resolve()
    paths = (
        KB_DIR / f"mitre_attack.{generation}.faiss",
        KB_DIR / f"mitre_attack.{generation}.metadata.json",
    )
    if any(path.resolve().parent != artifact_dir for path in paths):
        raise ArtifactError("FAISS generation path escapes the artifact directory.")
    return paths


def _validate_generation(
    faiss_module,
    generation,
    documents,
    model_name,
):
    index_path, metadata_path = _generation_paths(generation)
    if not index_path.exists() or not metadata_path.exists():
        raise ArtifactError(f"Generation is incomplete: {generation}")
    index = _read_index(faiss_module, index_path)
    metadata = _read_json(metadata_path, "FAISS metadata")
    if not isinstance(metadata, dict):
        raise ArtifactError("FAISS metadata must be a JSON object.")
    if metadata.get("generation") != generation:
        raise ArtifactError("FAISS metadata generation does not match its filename.")
    return validate_index(index, documents, metadata, faiss_module, model_name)


def _read_manifest():
    if not MANIFEST_PATH.exists():
        raise ArtifactError(f"FAISS manifest is missing: {MANIFEST_PATH}")
    try:
        manifest = _read_json(MANIFEST_PATH, "FAISS manifest")
    except ArtifactError as error:
        raise ArtifactError(
            f"FAISS manifest is malformed or unreadable: {MANIFEST_PATH}"
        ) from error
    if not isinstance(manifest, dict):
        raise ArtifactError("FAISS manifest is malformed: expected a JSON object.")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactError("FAISS manifest has an unsupported schema.")
    generation = manifest.get("generation")
    _generation_paths(generation)
    previous_generation = manifest.get("previous_generation")
    if previous_generation is not None:
        _generation_paths(previous_generation)
        if previous_generation == generation:
            raise ArtifactError(
                "FAISS manifest is malformed: active and previous generations match."
            )
    return generation, previous_generation


def load_validated_index(faiss_module, documents, model_name=MODEL_NAME):
    """Load the active generation, then only its designated previous generation."""
    generation, previous_generation = _read_manifest()
    try:
        return _validate_generation(
            faiss_module, generation, documents, model_name
        )
    except ArtifactError as active_error:
        if previous_generation is None:
            raise ArtifactError(
                f"Active FAISS generation {generation} is invalid and no "
                f"designated fallback exists: {active_error}"
            ) from active_error
        try:
            index = _validate_generation(
                faiss_module, previous_generation, documents, model_name
            )
        except ArtifactError as fallback_error:
            raise ArtifactError(
                f"Active FAISS generation {generation} is invalid; designated "
                f"fallback {previous_generation} is also invalid: "
                f"{fallback_error}"
            ) from fallback_error
        print(
            f"Active FAISS generation {generation} is invalid; "
            f"using designated fallback {previous_generation}."
        )
        return index


def _metadata(documents, dimension, model_name, generation):
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "generation": generation,
        "corpus_fingerprint": corpus_fingerprint(documents),
        "model_name": model_name,
        "vector_dimension": int(dimension),
        "document_count": len(documents),
        "index_type": EXPECTED_INDEX_TYPE,
        "metric_type": EXPECTED_METRIC_TYPE,
    }


def _write_temp_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_or_reuse_index(
    faiss_module,
    documents,
    model,
    model_name=MODEL_NAME,
):
    """Reuse a valid generation or publish a new one through the manifest."""
    previous_generation = None
    if MANIFEST_PATH.exists():
        active_generation, designated_previous = _read_manifest()
        try:
            return _validate_generation(
                faiss_module, active_generation, documents, model_name
            )
        except ArtifactError as active_error:
            print(
                f"Active FAISS generation {active_generation} is unavailable: "
                f"{active_error}"
            )
            if designated_previous is not None:
                try:
                    _validate_generation(
                        faiss_module,
                        designated_previous,
                        documents,
                        model_name,
                    )
                except ArtifactError as fallback_error:
                    print(
                        f"Designated fallback {designated_previous} is "
                        f"unavailable: {fallback_error}"
                    )
                else:
                    previous_generation = designated_previous
                    print(
                        f"Preserving designated fallback generation: "
                        f"{designated_previous}"
                    )
            print("Building a validated FAISS index...")
    else:
        print(f"Existing retrieval artifacts are unavailable: "
              f"FAISS manifest is missing: {MANIFEST_PATH}")
        print("Building a validated FAISS index...")

    if not documents:
        raise ValueError("The MITRE document collection is empty.")

    texts = [document.get("text", "") for document in documents]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("Embedding output does not match the document collection.")

    index = faiss_module.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    generation = uuid.uuid4().hex
    metadata = _metadata(
        documents, embeddings.shape[1], model_name, generation
    )
    validate_index(index, documents, metadata, faiss_module, model_name)

    KB_DIR.mkdir(parents=True, exist_ok=True)
    final_index, final_metadata = _generation_paths(generation)
    index_fd, index_tmp_name = tempfile.mkstemp(
        prefix=f"{final_index.name}.", suffix=".tmp", dir=KB_DIR
    )
    metadata_fd, metadata_tmp_name = tempfile.mkstemp(
        prefix=f"{final_metadata.name}.", suffix=".tmp", dir=KB_DIR
    )
    manifest_fd, manifest_tmp_name = tempfile.mkstemp(
        prefix=f"{MANIFEST_PATH.name}.", suffix=".tmp", dir=KB_DIR
    )
    os.close(index_fd)
    os.close(metadata_fd)
    os.close(manifest_fd)
    index_tmp = Path(index_tmp_name)
    metadata_tmp = Path(metadata_tmp_name)
    manifest_tmp = Path(manifest_tmp_name)

    try:
        faiss_module.write_index(index, str(index_tmp))
        written_index = _read_index(faiss_module, index_tmp)
        validate_index(
            written_index, documents, metadata, faiss_module, model_name
        )
        _write_temp_json(metadata_tmp, metadata)
        os.replace(index_tmp, final_index)
        os.replace(metadata_tmp, final_metadata)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generation": generation,
        }
        if previous_generation is not None:
            manifest["previous_generation"] = previous_generation
        _write_temp_json(manifest_tmp, manifest)
        os.replace(manifest_tmp, MANIFEST_PATH)
    except (OSError, ValueError, RuntimeError) as error:
        raise ArtifactError("Could not safely publish FAISS artifacts.") from error
    finally:
        index_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)

    print("Published FAISS generation:", generation)
    return index
