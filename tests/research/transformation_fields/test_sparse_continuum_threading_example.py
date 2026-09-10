# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from research.transformation_fields.examples.sparse_continuum_threading import (
    Config,
    build_field,
    build_robot,
    build_scene,
    coarse_dense_shape_error,
    verify_final,
)

pytestmark = pytest.mark.unit


def test_continuum_dense_transfer_preserves_inverse_and_section_rigidity_without_retraining():
    config = replace(Config(), optimization_sections=12, surface_samples=4, dense_sections=31, dense_surface_samples=8)
    device, dtype = torch.device("cpu"), torch.float64
    scene = build_scene(config, device=device, dtype=dtype)
    coarse = build_robot(config.optimization_sections, config.surface_samples, scene, config)
    dense = build_robot(config.dense_sections, config.dense_surface_samples, scene, config)
    field_model = build_field(config, device=device, dtype=dtype)
    with torch.no_grad():
        field_model.latent_coordinates.zero_()

    coarse_report, coarse_final = verify_final(field_model, coarse, scene)
    dense_report, dense_final = verify_final(field_model, dense, scene)

    assert config.live is False
    assert dense.point_count > coarse.point_count
    assert coarse_report["inverse_reconstruction_error"] < 1e-12
    assert dense_report["inverse_reconstruction_error"] < 1e-12
    assert coarse_report["cross_section_rigidity_error"] < 1e-12
    assert dense_report["cross_section_rigidity_error"] < 1e-12
    assert coarse_dense_shape_error(coarse, coarse_final, dense, dense_final) < 1e-12
