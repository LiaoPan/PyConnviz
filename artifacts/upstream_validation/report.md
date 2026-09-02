# PyConnviz upstream accuracy validation

**Overall:** PASS

Required checks: 38 passed, 0 failed, 38 total.
Diagnostics: 0 matched, 2 observations differed, 2 total.

Machine-readable expected/observed values are in `report.json`.

## Environment

- matplotlib: `3.11.1`
- mne: `1.12.1`
- mne-connectivity: `0.9.0`
- nilearn: `0.14.0`
- numpy: `2.5.2`
- pyconnviz: `0.1.0`
- python: `3.13.9`

## Upstream source snapshots

- [nilearn_plot_probabilistic_atlas_extraction.py](https://nilearn.github.io/stable/_downloads/c4ffee04091cacf1483ed5d324ce9c8e/plot_probabilistic_atlas_extraction.py) — SHA-256 `9093688dab31c79f1e65de75e7a45c7506766f1ff4ac57dbaf152b79443f3d78`, 3637 bytes
- [nilearn_plot_sphere_based_connectome.py](https://nilearn.github.io/stable/_downloads/cb064f9ac4369c8f91e970b8f1ae7965/plot_sphere_based_connectome.py) — SHA-256 `46ace49488c1c185f1a7a6d1a065d9662dc2935251de8bc85486ab59ebdde7eb`, 11882 bytes
- [mne_connectivity_connectivity_classes.py](https://mne.tools/mne-connectivity/stable/_downloads/980054063d6073a3d7942ac239432571/connectivity_classes.py) — SHA-256 `f1e94a1a07ff17ac1bf00072aa020a53c87b902eec30a1040b9824a1e2b858bd`, 4380 bytes
- [mne_connectivity_compare_coherency_methods.py](https://mne.tools/mne-connectivity/stable/_downloads/584a0ffc2ce7de5433ac678cbf086b83/compare_coherency_methods.py) — SHA-256 `2776ff3605bf202112593388bad64d1b5fdf562be478d982e136d1f870cb813f`, 20543 bytes
- [mne_connectivity_sensor_connectivity.py](https://mne.tools/mne-connectivity/stable/_downloads/c97704849885b49e703ad1a2fa96a3dd/sensor_connectivity.py) — SHA-256 `f74f4d86c01acc4659d827524d9d6fc8718dedc820f1439d62b3b084b9afe3ee`, 2228 bytes

## Scope

- Surface placement accuracy claimed: `False`
- Surface visualization audited: `True`
- Individual surface-registration truth claimed: `False`
- Product code modified by this audit: `True`

## Required failures

None.

## Diagnostic observations

### [DIFF] `nilearn.native_html_edges`

- Expected: `list with 152 entries; see report.json`
- Observed: `list with 148 entries; see report.json`
- Details: `{"nilearn_selected_count": 152, "nilearn_threshold": 0.30585996806038335, "pyconnviz_selected_count": 148, "semantics": "full dense matrix including diagonal and duplicate triangles"}`

### [DIFF] `nilearn.power.estimator_convergence`

- Expected: `"GraphicalLassoCV completes without convergence warnings"`
- Observed: `["graphical_lasso: did not converge after 100 iteration: dual gap: 7.081e-02"]`
- Details: `{"scope": "upstream estimator diagnostic; edge/render parity remains exact for the returned public covariance_ matrix"}`

## All checks

| Status | Role | Check | Maximum absolute error |
|---|---|---|---:|
| PASS | required | `mne.public_contract` | — |
| PASS | required | `mne.dense_values_read` | 0.0 |
| PASS | required | `mne.symmetric_dense_auto` | 0.0 |
| PASS | required | `mne.auto_edges` | — |
| PASS | required | `mne.auto_strength` | 5.551115123125783e-17 |
| PASS | required | `mne.explicit_lower_matrix` | 0.0 |
| PASS | required | `mne.explicit_lower_edges` | — |
| PASS | required | `mne.explicit_lower_strength` | 5.551115123125783e-17 |
| PASS | required | `mne.coh.auto_symmetric_matrix` | 0.0 |
| PASS | required | `mne.coh.circle_edge_identity` | — |
| PASS | required | `mne.coh.circle_pixel_parity` | 0.0 |
| PASS | required | `mne.plv.auto_symmetric_matrix` | 0.0 |
| PASS | required | `mne.plv.circle_edge_identity` | — |
| PASS | required | `mne.plv.circle_pixel_parity` | 0.0 |
| PASS | required | `nilearn.real_data_pipeline` | — |
| PASS | required | `nilearn.matrix_preserved` | 0.0 |
| PASS | required | `nilearn.node_order` | — |
| PASS | required | `nilearn.mni_coordinates` | 3.051757815342171e-06 |
| PASS | required | `nilearn.scientific_top20_edges` | — |
| PASS | required | `nilearn.native_static_edges` | — |
| DIFF | diagnostic | `nilearn.native_html_edges` | — |
| PASS | required | `nilearn.node_strength` | 1.7763568394002505e-15 |
| PASS | required | `nilearn.backend_edge_identity` | — |
| PASS | required | `nilearn.matched_static_pixels` | 0.0 |
| PASS | required | `nilearn.artifacts` | — |
| PASS | required | `nilearn.power.real_data_pipeline` | — |
| PASS | required | `nilearn.power.matrix_preserved` | 0.0 |
| PASS | required | `nilearn.power.percentile_edges` | — |
| PASS | required | `nilearn.power.native_static_edges` | — |
| DIFF | diagnostic | `nilearn.power.estimator_convergence` | — |
| PASS | required | `nilearn.power.node_strength` | 7.105427357601002e-15 |
| PASS | required | `nilearn.power.backend_edge_identity` | — |
| PASS | required | `nilearn.power.matched_static_pixels` | 0.0 |
| PASS | required | `nilearn.power.artifacts` | — |
| PASS | required | `surface.fsaverage_projection` | 9.497894247794829 |
| PASS | required | `surface.matrix_subset_preserved` | 0.0 |
| PASS | required | `surface.top40_edges` | — |
| PASS | required | `surface.node_strength` | 8.881784197001252e-16 |
| PASS | required | `surface.backend_edge_identity` | — |
| PASS | required | `surface.artifacts` | — |
