import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import retrieval_artifacts as artifacts


class Embeddings:
    def __init__(self, rows):
        self.ndim = 2
        self.shape = (len(rows), len(rows[0]))

    def astype(self, _dtype):
        return self


class Model:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def encode(self, texts, **_kwargs):
        self.calls += 1
        if len(texts) != len(self.rows):
            raise AssertionError("Unexpected embedding request")
        return Embeddings(self.rows)


class IndexFlatL2:
    def __init__(self, dimension):
        self.d = dimension
        self.ntotal = 0
        self.metric_type = 1

    def add(self, embeddings):
        self.ntotal = embeddings.shape[0]


class WrongMetricIndex(IndexFlatL2):
    def __init__(self, dimension):
        super().__init__(dimension)
        self.metric_type = 2


class WrongTypeIndex(IndexFlatL2):
    pass


class FakeFaiss:
    METRIC_L2 = 1
    IndexFlatL2 = IndexFlatL2

    @staticmethod
    def write_index(index, path):
        Path(path).write_text(
            json.dumps(
                {
                    "type": index.__class__.__name__,
                    "dimension": index.d,
                    "count": index.ntotal,
                    "metric": index.metric_type,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def read_index(path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        index_types = {
            "IndexFlatL2": IndexFlatL2,
            "WrongMetricIndex": WrongMetricIndex,
            "WrongTypeIndex": WrongTypeIndex,
        }
        index = index_types[data["type"]](data["dimension"])
        index.ntotal = data["count"]
        index.metric_type = data.get("metric", index.metric_type)
        return index


class RetrievalArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.patches = patch.multiple(
            artifacts,
            KB_DIR=self.root,
            MANIFEST_PATH=self.root / "mitre_attack.current.json",
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)
        self.addCleanup(self.temp_dir.cleanup)
        self.documents = [{"text": "one"}, {"text": "two"}]
        self.rows = [[1.0, 2.0], [3.0, 4.0]]

    def build(self, documents=None, model=None):
        return artifacts.build_or_reuse_index(
            FakeFaiss,
            documents or self.documents,
            model or Model(self.rows),
        )

    def active_generation(self):
        manifest = json.loads(
            (self.root / "mitre_attack.current.json").read_text()
        )
        return manifest["generation"]

    def generation_paths(self):
        generation = self.active_generation()
        return (
            self.root / f"mitre_attack.{generation}.faiss",
            self.root / f"mitre_attack.{generation}.metadata.json",
        )

    def set_manifest(self, generation, previous_generation=None):
        manifest = {"schema_version": 1, "generation": generation}
        if previous_generation is not None:
            manifest["previous_generation"] = previous_generation
        (self.root / "mitre_attack.current.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def duplicate_active_generation(self, generation):
        index_path, metadata_path = self.generation_paths()
        new_index = self.root / f"mitre_attack.{generation}.faiss"
        new_metadata = self.root / f"mitre_attack.{generation}.metadata.json"
        new_index.write_bytes(index_path.read_bytes())
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["generation"] = generation
        new_metadata.write_text(json.dumps(metadata), encoding="utf-8")

    def test_fresh_creation_and_metadata(self):
        index = self.build()
        index_path, metadata_path = self.generation_paths()
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(index.ntotal, 2)
        self.assertTrue(index_path.exists())
        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["index_type"], "IndexFlatL2")
        self.assertEqual(metadata["metric_type"], "METRIC_L2")

    def test_valid_reuse_does_not_encode(self):
        self.build()
        model = Model(self.rows)
        self.build(model=model)
        self.assertEqual(model.calls, 0)

    def test_stale_corpus_rebuilds(self):
        self.build()
        model = Model(self.rows)
        self.build(documents=[{"text": "changed"}, {"text": "two"}], model=model)
        self.assertEqual(model.calls, 1)

    def test_valid_active_followed_by_rebuild_does_not_designate_incompatible_active(self):
        self.build()
        old_generation = self.active_generation()
        self.build(
            documents=[{"text": "changed"}, {"text": "two"}],
            model=Model(self.rows),
        )
        manifest = json.loads(
            (self.root / "mitre_attack.current.json").read_text()
        )
        self.assertNotEqual(manifest["generation"], old_generation)
        self.assertNotIn("previous_generation", manifest)

    def test_model_and_dimension_mismatch_are_rejected(self):
        self.build()
        _, metadata_path = self.generation_paths()
        metadata = json.loads(metadata_path.read_text())
        metadata["model_name"] = "different-model"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_validated_index(
                FakeFaiss, self.documents, "all-MiniLM-L6-v2"
            )

        metadata["model_name"] = "all-MiniLM-L6-v2"
        metadata["vector_dimension"] = 99
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_invalid_generation_identifiers_are_rejected_before_path_access(self):
        outside = self.root.parent / "outside.faiss"
        for generation in (
            "../../../outside",
            "/tmp/outside",
            "abc/def",
            "",
            "not-a-uuid",
            "A" * 32,
        ):
            with self.subTest(generation=generation):
                with self.assertRaisesRegex(
                    artifacts.ArtifactError, "generation identifier"
                ):
                    artifacts._generation_paths(generation)
                self.assertFalse(outside.exists())

    def test_valid_generated_identifier_stays_inside_artifact_directory(self):
        paths = artifacts._generation_paths("a" * 32)
        self.assertTrue(all(path.parent == self.root for path in paths))

    def test_missing_manifest_is_distinct(self):
        with self.assertRaisesRegex(artifacts.ArtifactError, "manifest is missing"):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_malformed_manifest_is_explicit_and_does_not_fallback(self):
        self.build()
        manifest = self.root / "mitre_attack.current.json"
        manifest.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(
            artifacts.ArtifactError, "manifest is malformed"
        ):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_corrupt_active_uses_designated_previous_generation(self):
        self.build()
        previous = self.active_generation()
        active = "b" * 32
        self.duplicate_active_generation(active)
        self.set_manifest(active, previous)
        active_index = self.root / f"mitre_attack.{active}.faiss"
        active_index.write_text("not-json", encoding="utf-8")
        self.assertEqual(
            artifacts.load_validated_index(FakeFaiss, self.documents).ntotal,
            2,
        )

    def test_corrupt_active_rebuild_preserves_valid_designated_fallback(self):
        self.build()
        previous = self.active_generation()
        active = "e" * 32
        self.duplicate_active_generation(active)
        self.set_manifest(active, previous)
        (self.root / f"mitre_attack.{active}.faiss").write_text(
            "not-json", encoding="utf-8"
        )
        self.build(model=Model(self.rows))
        manifest = json.loads(
            (self.root / "mitre_attack.current.json").read_text()
        )
        self.assertNotEqual(manifest["generation"], active)
        self.assertEqual(manifest["previous_generation"], previous)

    def test_incompatible_active_rebuild_preserves_valid_designated_fallback(self):
        self.build()
        previous = self.active_generation()
        active = "f" * 32
        self.duplicate_active_generation(active)
        active_metadata = self.root / f"mitre_attack.{active}.metadata.json"
        metadata = json.loads(active_metadata.read_text())
        metadata["model_name"] = "different-model"
        active_metadata.write_text(json.dumps(metadata), encoding="utf-8")
        self.set_manifest(active, previous)
        self.build(model=Model(self.rows))
        manifest = json.loads(
            (self.root / "mitre_attack.current.json").read_text()
        )
        self.assertNotEqual(manifest["generation"], active)
        self.assertEqual(manifest["previous_generation"], previous)

    def test_rebuild_drops_unusable_designated_fallback(self):
        self.build()
        previous = self.active_generation()
        active = "1" * 32
        self.duplicate_active_generation(active)
        self.set_manifest(active, previous)
        (self.root / f"mitre_attack.{active}.faiss").write_text(
            "not-json", encoding="utf-8"
        )
        (self.root / f"mitre_attack.{previous}.faiss").write_text(
            "not-json", encoding="utf-8"
        )
        self.build(model=Model(self.rows))
        manifest = json.loads(
            (self.root / "mitre_attack.current.json").read_text()
        )
        self.assertNotIn("previous_generation", manifest)

    def test_malformed_manifest_blocks_rebuild(self):
        self.build()
        (self.root / "mitre_attack.current.json").write_text(
            "{", encoding="utf-8"
        )
        model = Model(self.rows)
        with self.assertRaisesRegex(
            artifacts.ArtifactError, "manifest is malformed"
        ):
            self.build(model=model)
        self.assertEqual(model.calls, 0)

    def test_arbitrary_stale_generation_is_not_selected(self):
        self.build()
        active = self.active_generation()
        stale = "c" * 32
        self.duplicate_active_generation(stale)
        self.set_manifest(active)
        self.assertEqual(
            artifacts.load_validated_index(FakeFaiss, self.documents).ntotal,
            2,
        )
        (self.root / f"mitre_attack.{active}.faiss").write_text(
            "not-json", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            artifacts.ArtifactError, "no designated fallback"
        ):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_no_safe_fallback_is_explicit(self):
        self.build()
        active = self.active_generation()
        (self.root / f"mitre_attack.{active}.faiss").write_text(
            "not-json", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            artifacts.ArtifactError, "no designated fallback"
        ):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_invalid_json_metadata_reaches_metadata_parser(self):
        self.build()
        index_path, metadata_path = self.generation_paths()
        metadata_path.write_text("{", encoding="utf-8")
        with patch.object(
            FakeFaiss, "read_index", wraps=FakeFaiss.read_index
        ) as read_index:
            with self.assertRaisesRegex(artifacts.ArtifactError, "metadata"):
                artifacts.load_validated_index(FakeFaiss, self.documents)
        read_index.assert_called_once_with(str(index_path))

    def test_wrong_structure_metadata_reaches_metadata_validation(self):
        self.build()
        index_path, metadata_path = self.generation_paths()
        metadata_path.write_text("[]", encoding="utf-8")
        with patch.object(
            FakeFaiss, "read_index", wraps=FakeFaiss.read_index
        ) as read_index:
            with self.assertRaisesRegex(
                artifacts.ArtifactError, "JSON object"
            ):
                artifacts.load_validated_index(FakeFaiss, self.documents)
        read_index.assert_called_once_with(str(index_path))

    def test_corrupt_index_with_valid_metadata_is_rejected(self):
        self.build()
        index_path, _ = self.generation_paths()
        index_path.write_text("not-json", encoding="utf-8")
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_wrong_metric_is_rejected(self):
        self.build()
        index_path, metadata_path = self.generation_paths()
        metadata = json.loads(metadata_path.read_text())
        index_path.write_text(
            json.dumps(
                {
                    "type": "WrongMetricIndex",
                    "dimension": 2,
                    "count": 2,
                    "metric": 2,
                }
            ),
            encoding="utf-8",
        )
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_wrong_index_type_is_rejected(self):
        self.build()
        index_path, _ = self.generation_paths()
        index_path.write_text(
            json.dumps(
                {
                    "type": "WrongTypeIndex",
                    "dimension": 2,
                    "count": 2,
                    "metric": 1,
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_failed_build_preserves_active_generation(self):
        self.build()
        old_manifest = (self.root / "mitre_attack.current.json").read_bytes()
        model = Model(self.rows)
        with patch.object(
            FakeFaiss,
            "write_index",
            side_effect=OSError("simulated write failure"),
        ):
            with self.assertRaises(artifacts.ArtifactError):
                self.build(
                    documents=[{"text": "changed"}, {"text": "two"}],
                    model=model,
                )
        self.assertEqual(
            (self.root / "mitre_attack.current.json").read_bytes(),
            old_manifest,
        )
        self.assertEqual(
            artifacts.load_validated_index(FakeFaiss, self.documents).ntotal,
            2,
        )

    def test_manifest_replacement_failure_preserves_active_generation(self):
        self.build()
        old_generation = self.active_generation()
        model = Model(self.rows)
        real_replace = artifacts.os.replace

        def fail_manifest_replace(source, destination):
            if Path(destination) == self.root / "mitre_attack.current.json":
                raise OSError("simulated interruption")
            return real_replace(source, destination)

        with patch.object(artifacts.os, "replace", side_effect=fail_manifest_replace):
            with self.assertRaises(artifacts.ArtifactError):
                self.build(
                    documents=[{"text": "changed"}, {"text": "two"}],
                    model=model,
                )
        self.assertEqual(self.active_generation(), old_generation)
        artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_missing_manifest_generation_falls_back_to_previous(self):
        self.build()
        generation = self.active_generation()
        (self.root / f"mitre_attack.{generation}.faiss").unlink()
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.load_validated_index(FakeFaiss, self.documents)

    def test_manifest_invalid_generation_falls_back_to_previous(self):
        self.build()
        old_generation = self.active_generation()
        old_index, old_metadata = self.generation_paths()
        new_generation = "d" * 32
        self.duplicate_active_generation(new_generation)
        self.set_manifest(new_generation, old_generation)
        new_index, new_metadata = self.generation_paths()
        new_index.write_text("not-json", encoding="utf-8")
        self.assertEqual(
            artifacts.load_validated_index(FakeFaiss, self.documents).ntotal,
            2,
        )
        self.assertTrue(old_index.exists())
        self.assertTrue(old_metadata.exists())
        self.assertNotEqual(old_generation, new_generation)

    def test_original_fingerprint_separator_is_preserved(self):
        digest = hashlib.sha256()
        for document in self.documents:
            canonical = json.dumps(
                document,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            digest.update(canonical.encode("utf-8"))
            digest.update(b"\\n")
        self.assertEqual(
            artifacts.corpus_fingerprint(self.documents), digest.hexdigest()
        )


class NotebookConsistencyTests(unittest.TestCase):
    def test_notebook_does_not_generate_runtime_script(self):
        notebook = json.loads(
            Path("notebooks/01_data_preprocessing_and_rag_setup.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "".join(notebook["cells"][136]["source"])
        self.assertIn("Runtime source of truth", source)
        self.assertNotIn("write_text", source)
        self.assertNotIn("script_code", source)

    def test_notebook_has_configurable_root_and_data_paths(self):
        notebook = json.loads(
            Path("notebooks/01_data_preprocessing_and_rag_setup.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "".join(notebook["cells"][1]["source"])
        self.assertIn("TRUSTSOC_RAW_DATA_DIR", source)
        self.assertIn("TRUSTSOC_PROCESSED_DATA_DIR", source)
        self.assertIn("PROJECT_ROOT", source)


class RetrievalContractTests(unittest.TestCase):
    def test_evaluation_scripts_use_the_canonical_artifact_loader(self):
        names = (
            "evaluate_retrieval.py",
            "evaluate_labeled_retrieval.py",
            "evaluate_query_expansion.py",
            "experiment_query_expansion.py",
        )
        for name in names:
            source = Path("scripts", name).read_text(encoding="utf-8")
            self.assertIn("load_validated_index", source)
            self.assertIn("MODEL_NAME", source)

    def test_runtime_retrieval_preserves_index_order_and_top_k(self):
        class FakeNumpy:
            @staticmethod
            def asarray(value, dtype=None):
                return value

        class QueryEmbedding:
            pass

        class Model:
            def encode(self, _queries, **_kwargs):
                return QueryEmbedding()

        class Index:
            def search(self, _embedding, count):
                self.count = count
                return [[0.1, 0.2]], [[1, 0]]

        fake_sentence_transformers = type(
            "SentenceTransformers", (), {"SentenceTransformer": object}
        )
        fake_pandas = type("Pandas", (), {})
        fake_faiss = type("Faiss", (), {})
        previous = {
            name: sys.modules.get(name)
            for name in ("numpy", "pandas", "faiss", "sentence_transformers")
        }
        sys.modules["numpy"] = FakeNumpy
        sys.modules["pandas"] = fake_pandas
        sys.modules["faiss"] = fake_faiss
        sys.modules["sentence_transformers"] = fake_sentence_transformers
        try:
            spec = importlib.util.spec_from_file_location(
                "demo_threat_intel_test_module",
                Path("scripts/demo_threat_intel.py"),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results = module.retrieve_mitre_evidence(
                "query",
                [
                    {"technique_id": "T0", "name": "zero"},
                    {"technique_id": "T1", "name": "one"},
                ],
                Model(),
                Index(),
                top_k=2,
            )
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.assertEqual(
            [item["technique_id"] for item in results], ["T1", "T0"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
