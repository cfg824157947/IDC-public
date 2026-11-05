from __future__ import annotations

from pathlib import Path

import pytest

from idc.canonical import canonical_config_hash, canonical_config_json, resolved_run_hashes
from idc.validation.task_paths import ExistingTaskOutputError, assert_fresh_task_namespace, task_namespace
from scripts.resolve_run_config import resolve


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_config_serialization_orders_keys_and_preserves_json_values():
    left = {"z": None, "b": False, "a": 1, "f": 1.0}
    right = {"f": 1.0, "a": 1, "b": False, "z": None}
    assert canonical_config_json(left) == canonical_config_json(right)
    assert canonical_config_json(left) == '{"a":1,"b":false,"f":1.0,"z":null}'
    assert canonical_config_hash(left) == canonical_config_hash(right)
    assert canonical_config_hash({"a": 1}) != canonical_config_hash({"a": 1.0})


def test_run_plan_fit_and_record_share_canonical_scientific_hash():
    config, execution = resolve(
        ROOT / "configs/scientific_publication.yaml",
        ROOT / "configs/execution_cpu.yaml",
    )
    hashes = resolved_run_hashes(config.to_dict(), execution)
    assert hashes["scientific_config_sha256"] == canonical_config_hash(config.to_dict())
    assert hashes["execution_config_sha256"] == canonical_config_hash(execution)
    assert hashes["run_config_sha256"] == canonical_config_hash(
        {"scientific_config": config.to_dict(), "execution_config": execution}
    )


def test_direct_and_sbatch_configuration_use_same_legacy_score_setting():
    direct_config, direct_execution = resolve(
        ROOT / "configs/scientific_publication.yaml",
        ROOT / "configs/execution_cpu.yaml",
    )
    sbatch_templates = [
        (ROOT / "submit/main_grid_cpu.sbatch").read_text(),
        (ROOT / "submit/main_grid_fold_cpu.sbatch").read_text(),
    ]
    for sbatch in sbatch_templates:
        assert "--scientific-config configs/scientific_publication.yaml" in sbatch
        assert "--execution-config configs/execution_cpu.yaml" in sbatch
        assert "include_legacy_gene_score" not in sbatch
        assert "legacy_reproduction" not in sbatch
        assert "#SBATCH --mem=40G" in sbatch
        assert "#SBATCH --cpus-per-task=4" in sbatch
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert f'export {name}="$THREADS"' in sbatch
    wrapper_config, wrapper_execution = resolve(
        ROOT / "configs/scientific_publication.yaml",
        ROOT / "configs/execution_cpu.yaml",
    )
    assert direct_config.gene_score.legacy_reproduction is False
    assert wrapper_config.gene_score.legacy_reproduction == direct_config.gene_score.legacy_reproduction
    assert wrapper_config.to_dict() == direct_config.to_dict()
    assert wrapper_execution == direct_execution
    assert wrapper_execution["slurm"]["memory"] == "40G"
    assert wrapper_execution["slurm"]["cpus_per_task"] == wrapper_execution["slurm"]["threads_per_task"] == 4


def test_task_namespaces_are_unique_and_existing_checkpoint_fails_closed(tmp_path):
    base = dict(analysis_class="PRIMARY", dataset_id="dset_a", fold_id=1, task_id=4)
    one = task_namespace(tmp_path, **base, run_id="checkpoint")
    other_fold = task_namespace(tmp_path, **{**base, "fold_id": 2, "task_id": 5})
    other_dataset = task_namespace(tmp_path, **{**base, "dataset_id": "dset_b"})
    other_class = task_namespace(tmp_path, **{**base, "analysis_class": "DESCRIPTIVE"})
    other_task = task_namespace(tmp_path, **{**base, "task_id": 9})
    other_run = task_namespace(tmp_path, **base, run_id="rerun_2")
    assert len({one, other_fold, other_dataset, other_class, other_task, other_run}) == 6
    assert_fresh_task_namespace(one)
    one.mkdir(parents=True)
    with pytest.raises(ExistingTaskOutputError):
        assert_fresh_task_namespace(one)
    assert_fresh_task_namespace(other_run)


def test_execution_thread_count_must_match_cpu_count(tmp_path):
    source = (ROOT / "configs/execution_cpu.yaml").read_text()
    bad = tmp_path / "execution.yaml"
    bad.write_text(source.replace("threads_per_task: 4", "threads_per_task: 2"))
    with pytest.raises(ValueError, match="threads_per_task"):
        resolve(ROOT / "configs/scientific_publication.yaml", bad)


def test_main_grid_plan_has_one_unique_namespaced_task_per_dataset_fold():
    import csv

    with (ROOT / "manifests/main_grid_fold_plan.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 180
    assert [int(row["task_id"]) for row in rows] == list(range(180))
    assert len({(row["dataset_id"], int(row["fold_id"])) for row in rows}) == 180
    assert len({row["result_jsonl"] for row in rows}) == 180
    assert len({row["review_bundle_dir"] for row in rows}) == 180
    assert len({row["run_config_sha256"] for row in rows}) == 1
    assert {int(row["fold_id"]) for row in rows} == {1, 2, 3, 4, 5}
    assert not any(row["preexisting_output_collision"] == "True" for row in rows)
    assert all(int(row["task_id"]) == 5 * int(row["manifest_index_zero_based"]) + int(row["fold_id"]) - 1 for row in rows)
    assert all(row["memory"] == "40G" and row["cpus"] == row["threads"] == "4" for row in rows)
