from benchmarks.provenance import _source, capture_immutable_provenance, capture_provenance


def test_source_fingerprint_covers_base_production_diff_and_harness():
    source = _source()

    assert len(source["base_git_commit"]) == 40
    for name in ("production_diff_sha256", "benchmark_sources_sha256", "measurement_source_sha256"):
        assert len(source[name]) == 64
        int(source[name], 16)


def test_provenance_retains_cuda_runtime_fields_even_without_cuda():
    provenance = capture_provenance("cuda")

    assert "torch_cuda_runtime" in provenance["versions"]
    assert "torch_cudnn" in provenance["versions"]
    assert "cuda_available" in provenance["device"]


def test_immutable_provenance_excludes_process_runtime_fields():
    provenance = capture_immutable_provenance()

    assert "source" in provenance
    assert "git" in provenance
    assert "process_id" not in provenance
    assert "device" not in provenance
    assert "threads" not in provenance
