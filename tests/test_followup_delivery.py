"""Regression tests for immutable delivery ZIP identity and atomic handoff."""
from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import pytest

from goldthread import build_tools, cli, ledger
from pipeline import dataset_build, length_gate, package
from pipeline.common import seqlength
from pipeline.common.config import DEFAULTS

_MANIFEST = "goldthread-train/data/MANIFEST.json"
_SCRIPT = "goldthread-train/autotrain.py"


class _Tokenizer:
    def count(self, text):
        return len(text)

    def encode(self, text, **_kwargs):
        return list(map(ord, text))

    def decode(self, ids, **_kwargs):
        return "".join(map(chr, ids))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["paths"]["data_dir"] = str(tmp_path / "workspace")
    cfg["paths"]["sessions_dir"] = str(tmp_path / "sessions")
    train = tmp_path / "workspace" / "03_train"
    train.mkdir(parents=True)
    counts = {}
    for split, project in (("train", "orchard"), ("eval", "garden")):
        name = f"sft_step_windows_sharegpt.{split}.jsonl"
        row = {
            "sample_id": split,
            "metadata": {"project_group": f"game::{project}"},
            "conversations": [
                {"from": "system", "value": "Keep a short inventory."},
                {"from": "human", "value": f"Record the apples in the {project}."},
                {"from": "gpt", "value": "<think>Count the two baskets.</think>\n\nTwo baskets."},
            ],
        }
        (train / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
        counts[name] = 1
    dataset_build.create_build(train, cfg, "a" * 16, {}, counts)
    tokenizer = _Tokenizer()
    monkeypatch.setattr(length_gate, "load_gate_tokenizer", lambda: (tokenizer, "fixture"))
    monkeypatch.setattr(seqlength, "find_tokenizer", lambda **_: (tokenizer, "fixture"))
    return cfg


def _pack(cfg, **kwargs):
    return build_tools.pack_delivery(cfg=cfg, reason="synthetic delivery regression", **kwargs)


def _manifest(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read(_MANIFEST))


def _repack(path, mutate):
    with zipfile.ZipFile(path) as zf:
        members = [(name, zf.read(name)) for name in zf.namelist()]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in mutate(members):
            zf.writestr(zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0)), payload)


def test_no_gate_history_cannot_pass_repack_and_copy(workspace, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: workspace)
    with monkeypatch.context() as m:
        m.setattr(length_gate, "load_gate_tokenizer", lambda: (None, "unavailable"))
        assert cli.main(["pack", "--no-gate", "--reason", "historical fixture", "--json"]) == 0
    old = json.loads(capsys.readouterr().out)
    old_zip = Path(old["zip_path"])
    old_bytes = old_zip.read_bytes()
    assert old["ok"] and old["gate"] is None
    assert not build_tools.gate_check(old_zip, cfg=workspace)["passed"]

    dest = tmp_path / "student.zip"
    assert cli.main(["pack", "--copy-to", str(dest), "--reason", "checked handoff", "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["package_hash"] == old["package_hash"]
    assert result["ok"] is False and result["zip_path"] is None and result["copied_to"] is None
    assert result["gate"]["resolved_path"] == str(old_zip.resolve())
    assert "length_gate_skipped" in {v["code"] for v in result["gate"]["violations"]}
    assert not dest.exists()
    assert old_zip.read_bytes() == old_bytes
    assert not list(old_zip.parent.glob(".staging-*"))


def test_publish_and_reuse_gate_report_published_path_and_copy_passes(workspace, tmp_path, monkeypatch):
    first = _pack(workspace, copy_to=tmp_path / "drop")
    assert first["ok"]
    archive = Path(first["zip_path"])
    assert first["gate"]["resolved_path"] == str(archive.resolve())
    assert Path(first["gate"]["target"]) == archive
    assert build_tools.gate_check(first["copied_to"], cfg=workspace)["passed"]

    # Different ZIP member order, compression, timestamps and generated_at are
    # not changes to the payload. Reuse the published bytes, not the candidate.
    def reserialize(members):
        for name, payload in reversed(members):
            if name == _MANIFEST:
                manifest = json.loads(payload)
                manifest["generated_at"] = "2000-01-01T00:00:00+00:00"
                payload = json.dumps(manifest, sort_keys=True).encode()
            yield name, payload

    _repack(archive, reserialize)
    original = archive.read_bytes()
    checked_paths = []
    real_gate = build_tools.gate_check

    def trace_gate(target, **kwargs):
        checked_paths.append(Path(target))
        return real_gate(target, **kwargs)

    monkeypatch.setattr(build_tools, "gate_check", trace_gate)
    second = _pack(workspace, copy_to=tmp_path / "student.zip")
    assert second["ok"] and second["reused_existing"]
    assert checked_paths == [archive]
    assert second["gate"]["resolved_path"] == str(archive.resolve())
    assert second["generated_at"] == _manifest(archive)["generated_at"]
    assert Path(second["copied_to"]).read_bytes() == archive.read_bytes() == original
    assert real_gate(second["copied_to"], cfg=workspace)["passed"]


@pytest.mark.parametrize("damage", ["changed", "missing", "extra", "corrupt"])
def test_same_hash_conflicts_never_publish_or_copy(workspace, tmp_path, damage):
    initial = _pack(workspace)
    assert initial["ok"]
    archive = Path(initial["zip_path"])
    if damage == "corrupt":
        # Keep the central directory readable but break an un-gated script's CRC.
        _repack(archive, lambda members: members)
        with zipfile.ZipFile(archive) as zf:
            info = zf.getinfo(_SCRIPT)
        with archive.open("r+b") as fp:
            fp.seek(info.header_offset + 30 + len(_SCRIPT))
            original_byte = fp.read(1)
            fp.seek(-1, 1)
            fp.write(bytes([original_byte[0] ^ 1]))
    else:
        def mutate(members):
            for name, payload in members:
                if name == _SCRIPT and damage == "missing":
                    continue
                yield name, payload + b"\n# fixture change\n" if name == _SCRIPT and damage == "changed" else payload
            if damage == "extra":
                yield "goldthread-train/extra.txt", b"fixture extra"

        _repack(archive, mutate)
    broken = archive.read_bytes()
    dest = tmp_path / "student.zip"
    assert build_tools.gate_check(archive, cfg=workspace)["passed"]
    with pytest.raises(ValueError, match="delivery ZIP"):
        _pack(workspace, copy_to=dest)
    with pytest.raises(ValueError, match="delivery ZIP"):
        package.build_delivery_zip_from_build(workspace)
    refused, published = ledger.read_ledger(workspace, tool="pack.delivery")
    assert published["op_id"] == initial["op_id"]
    assert refused["status"] == "refused" and refused["affected"]["pack"] is None
    assert refused["params"]["candidate_pack"] == initial["package_hash"]
    assert refused["params"]["requested_copy_to"] == str(dest)
    assert archive.read_bytes() == broken
    assert not dest.exists()
    assert not list(archive.parent.glob(".*.partial"))
    assert not list(archive.parent.glob(".staging-*"))


def test_parent_is_immutable_in_cli_and_lower_level(workspace, monkeypatch, capsys):
    first = _pack(workspace, profile="qwen-xml")
    assert first["ok"]
    child = _pack(workspace, profile="hermes-json")
    assert child["ok"]
    root_zip, child_zip = Path(first["zip_path"]), Path(child["zip_path"])
    root_bytes, child_bytes = root_zip.read_bytes(), child_zip.read_bytes()
    assert _manifest(root_zip)["parent_pack"] is None
    assert _manifest(child_zip)["parent_pack"] == first["package_hash"]

    monkeypatch.setattr(cli, "load_config", lambda: workspace)
    assert cli.main(["pack", "--profile", "qwen-xml", "--parent", child["package_hash"], "--json"]) == 1
    assert "Conflicting parent_pack" in json.loads(capsys.readouterr().out)["error"]
    refused, *published = ledger.read_ledger(workspace, tool="pack.delivery")
    assert len(published) == 2
    assert refused["status"] == "refused" and refused["affected"]["pack"] is None
    assert refused["params"]["parent_pack"] == child["package_hash"]

    with pytest.raises(ValueError, match="Conflicting parent_pack"):
        package.build_delivery_zip_from_build(workspace, profile="qwen-xml", parent_pack=child["package_hash"])
    for packer in (_pack, package.build_delivery_zip_from_build):
        repeat = packer(workspace, profile="hermes-json", parent_pack=first["zip_name"])
        assert repeat["reused_existing"] and repeat["parent_pack"] == first["package_hash"]
        assert root_zip.read_bytes() == root_bytes and child_zip.read_bytes() == child_bytes


def test_copy_to_same_file_is_a_noop(workspace, monkeypatch):
    packed = _pack(workspace)
    archive = Path(packed["zip_path"])
    before = archive.read_bytes()

    def unexpected_copy(*args):
        pytest.fail("copying a file to itself")

    monkeypatch.setattr(build_tools.shutil, "copy2", unexpected_copy)
    repeated = _pack(workspace, copy_to=archive.parent)
    assert repeated["ok"] and repeated["copied_to"] == str(archive.resolve())
    assert archive.read_bytes() == before


@pytest.mark.parametrize("failure", ["copy", "replace"])
def test_copy_failure_keeps_destination_and_cleans_temp(workspace, tmp_path, monkeypatch, failure):
    first = _pack(workspace)
    archive = Path(first["zip_path"])
    original = archive.read_bytes()
    dest_dir = tmp_path / "drop"
    dest_dir.mkdir()
    dest = dest_dir / "student.zip"
    dest.write_bytes(b"previous handoff")

    if failure == "copy":
        def fail_copy(src, target):
            assert Path(src) == archive
            assert target.parent == dest_dir and target != dest
            target.write_bytes(b"partial write")
            raise OSError("copy blocked")

        monkeypatch.setattr(build_tools.shutil, "copy2", fail_copy)
    else:
        real_replace = Path.replace

        def fail_replace(src, target):
            if target == dest:
                assert src.parent == dest_dir and src != dest
                assert src.read_bytes() == original
                raise OSError("replace blocked")
            return real_replace(src, target)

        monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match=f"{failure} blocked"):
        _pack(workspace, copy_to=dest)
    assert dest.read_bytes() == b"previous handoff"
    assert list(dest_dir.iterdir()) == [dest]
    assert archive.read_bytes() == original
    assert not list(archive.parent.glob(".staging-*"))
    failed, published = ledger.read_ledger(workspace, tool="pack.delivery")
    assert published["op_id"] == first["op_id"]
    assert failed["status"] == "reused_copy_failed" and failed["affected"]["pack"] is None
    assert failed["params"]["candidate_pack"] == first["package_hash"]


@pytest.mark.parametrize("failure", ["copy", "metadata"])
def test_failure_after_new_publication_is_recorded_once(workspace, tmp_path, monkeypatch, failure):
    delivery = tmp_path / "workspace" / "04_release" / "delivery"
    dest = tmp_path / "student.zip"
    dest.write_bytes(b"previous handoff")
    original_error = OSError(f"{failure} blocked")
    published_bytes = {}

    def fail_after_publish():
        archive, = delivery.glob("goldthread-train-*.zip")
        published_bytes[archive] = archive.read_bytes()
        raise original_error

    if failure == "copy":
        def fail_copy(src, target):
            target.write_bytes(b"partial copy")
            fail_after_publish()

        monkeypatch.setattr(build_tools.shutil, "copy2", fail_copy)
    else:
        real_replace = Path.replace

        def fail_replace(src, target):
            if target == delivery / "export_profile.json":
                fail_after_publish()
            return real_replace(src, target)

        monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError) as caught:
        _pack(workspace, copy_to=dest)
    assert caught.value is original_error
    (archive, original), = published_bytes.items()
    assert archive.read_bytes() == original
    assert build_tools.gate_check(archive, cfg=workspace)["passed"]
    assert dest.read_bytes() == b"previous handoff"
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(delivery.glob(".staging-*"))
    event, = ledger.read_ledger(workspace, tool="pack.delivery")
    assert event["status"] == f"published_{failure}_failed"
    manifest = _manifest(archive)
    assert event["affected"]["pack"] == manifest["package_hash"]
    assert event["affected"]["build_id"] == manifest["build_id"]
    assert event["params"]["requested_copy_to"] == str(dest)
    assert event["params"]["copy_to"] is None
    assert event["params"]["error"] == str(original_error)
