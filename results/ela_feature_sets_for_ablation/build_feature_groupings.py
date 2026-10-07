#!/usr/bin/env python3
"""Build exhaustive hierarchical ELA feature groupings for ablation studies."""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def fs(*features: str) -> list[str]:
    return list(features)


COMMON_ATOMS = {
    "disp_diff_mean": fs(*[f"disp.diff_mean_{q}" for q in ("02", "05", "10", "25")]),
    "disp_diff_median": fs(*[f"disp.diff_median_{q}" for q in ("02", "05", "10", "25")]),
    "disp_ratio_mean": fs(*[f"disp.ratio_mean_{q}" for q in ("02", "05", "10", "25")]),
    "disp_ratio_median": fs(*[f"disp.ratio_median_{q}" for q in ("02", "05", "10", "25")]),
    "distribution_shape": fs("ela_distr.kurtosis", "ela_distr.skewness"),
    "distribution_peaks": fs("ela_distr.number_of_peaks"),
    "level_discriminability": fs(*[f"ela_level.lda_qda_{q}" for q in ("10", "25", "50")]),
    "level_lda_error": fs(*[f"ela_level.mmce_lda_{q}" for q in ("10", "25", "50")]),
    "level_qda_error": fs(*[f"ela_level.mmce_qda_{q}" for q in ("10", "25", "50")]),
    "meta_linear_fit": fs("ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept"),
    "meta_linear_coefficients": fs("ela_meta.lin_simple.coef.max", "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_simple.coef.min"),
    "meta_nonlinear_fit": fs("ela_meta.lin_w_interact.adj_r2", "ela_meta.quad_simple.adj_r2", "ela_meta.quad_w_interact.adj_r2"),
    "meta_conditioning": fs("ela_meta.quad_simple.cond"),
    "ic_entropy": fs("ic.h_max", "ic.m0"),
    "ic_epsilon": fs("ic.eps_max", "ic.eps_ratio", "ic.eps_s"),
    "nbc_fitness_distance": fs("nbc.dist_ratio.coeff_var", "nbc.nb_fitness.cor"),
    "nbc_neighbor_ratios": fs("nbc.nn_nb.mean_ratio", "nbc.nn_nb.sd_ratio"),
    "nbc_neighbor_correlation": fs("nbc.nn_nb.cor"),
    "pca_global_variance": fs("pca.expl_var.cor_init"),
    "pca_pc1_variance": fs("pca.expl_var_PC1.cor_init", "pca.expl_var_PC1.cor_x", "pca.expl_var_PC1.cov_init", "pca.expl_var_PC1.cov_x"),
}

BBOB_ATOMS = {
    "disp_difference": fs("disp.diff_mean_02"),
    "disp_ratio": fs("disp.ratio_mean_02"),
    "distribution_shape": fs("ela_distr.kurtosis", "ela_distr.skewness"),
    "distribution_peaks": fs("ela_distr.number_of_peaks"),
    "level_discriminability_10": fs("ela_level.lda_qda_10"),
    "level_discriminability_25_50": fs("ela_level.lda_qda_25", "ela_level.lda_qda_50"),
    "level_lda_error_10_25": fs("ela_level.mmce_lda_10", "ela_level.mmce_lda_25"),
    "level_lda_error_50": fs("ela_level.mmce_lda_50"),
    "level_qda_error_10_25": fs("ela_level.mmce_qda_10", "ela_level.mmce_qda_25"),
    "level_qda_error_50": fs("ela_level.mmce_qda_50"),
    "meta_linear_fit": fs("ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept"),
    "meta_linear_coefficients": fs("ela_meta.lin_simple.coef.max", "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_simple.coef.min"),
    "meta_interaction_fit": fs("ela_meta.lin_w_interact.adj_r2"),
    "meta_quadratic_fit": fs("ela_meta.quad_simple.adj_r2", "ela_meta.quad_w_interact.adj_r2"),
    "meta_conditioning": fs("ela_meta.quad_simple.cond"),
    "ic_entropy": fs("ic.h_max", "ic.m0"),
    "ic_epsilon": fs("ic.eps_max"),
    "nbc_fitness": fs("nbc.nb_fitness.cor"),
    "nbc_neighbor_ratios": fs("nbc.nn_nb.mean_ratio", "nbc.nn_nb.sd_ratio"),
    "pca_global_variance": fs("pca.expl_var.cor_init"),
}


def combine(atoms: dict[str, list[str]], name: str, *keys: str) -> tuple[str, list[str]]:
    return name, [feature for key in keys for feature in atoms[key]]


def common_partitions() -> dict[str, dict[str, list[str]]]:
    a = COMMON_ATOMS
    p20 = dict(combine(a, key, key) for key in a)
    p15 = dict([
        combine(a, "disp_diff_mean", "disp_diff_mean"), combine(a, "disp_diff_median", "disp_diff_median"),
        combine(a, "disp_ratio_mean", "disp_ratio_mean"), combine(a, "disp_ratio_median", "disp_ratio_median"),
        combine(a, "objective_distribution", "distribution_shape", "distribution_peaks"),
        combine(a, "level_discriminability", "level_discriminability"), combine(a, "level_lda_error", "level_lda_error"),
        combine(a, "level_qda_error", "level_qda_error"), combine(a, "meta_linear_fit", "meta_linear_fit"),
        combine(a, "meta_linear_coefficients", "meta_linear_coefficients"),
        combine(a, "meta_nonlinearity", "meta_nonlinear_fit", "meta_conditioning"),
        combine(a, "information_content", "ic_entropy", "ic_epsilon"),
        combine(a, "nbc_fitness_distance", "nbc_fitness_distance"),
        combine(a, "nbc_neighbor_structure", "nbc_neighbor_ratios", "nbc_neighbor_correlation"),
        combine(a, "pca_variance_structure", "pca_global_variance", "pca_pc1_variance"),
    ])
    p10 = dict([
        combine(a, "dispersion_differences", "disp_diff_mean", "disp_diff_median"),
        combine(a, "dispersion_ratios", "disp_ratio_mean", "disp_ratio_median"),
        combine(a, "objective_distribution", "distribution_shape", "distribution_peaks"),
        combine(a, "level_discriminability", "level_discriminability"),
        combine(a, "level_classification_error", "level_lda_error", "level_qda_error"),
        combine(a, "meta_linear", "meta_linear_fit", "meta_linear_coefficients"),
        combine(a, "meta_nonlinearity", "meta_nonlinear_fit", "meta_conditioning"),
        combine(a, "information_content", "ic_entropy", "ic_epsilon"),
        combine(a, "nearest_better_geometry", "nbc_fitness_distance", "nbc_neighbor_ratios", "nbc_neighbor_correlation"),
        combine(a, "pca_variance_structure", "pca_global_variance", "pca_pc1_variance"),
    ])
    p5 = dict([
        combine(a, "rank_and_value_concentration", "disp_diff_mean", "disp_diff_median", "disp_ratio_mean", "disp_ratio_median", "distribution_shape", "distribution_peaks"),
        combine(a, "level_set_separability", "level_discriminability", "level_lda_error", "level_qda_error"),
        combine(a, "global_model_structure", "meta_linear_fit", "meta_linear_coefficients", "meta_nonlinear_fit", "meta_conditioning"),
        combine(a, "ruggedness_and_information", "ic_entropy", "ic_epsilon"),
        combine(a, "local_and_global_geometry", "nbc_fitness_distance", "nbc_neighbor_ratios", "nbc_neighbor_correlation", "pca_global_variance", "pca_pc1_variance"),
    ])
    return {"5": p5, "10": p10, "15": p15, "20": p20}


def bbob_partitions() -> dict[str, dict[str, list[str]]]:
    a = BBOB_ATOMS
    p20 = dict(combine(a, key, key) for key in a)
    p15 = dict([
        combine(a, "dispersion", "disp_difference", "disp_ratio"),
        combine(a, "distribution_shape", "distribution_shape"), combine(a, "distribution_peaks", "distribution_peaks"),
        combine(a, "level_discriminability", "level_discriminability_10", "level_discriminability_25_50"),
        combine(a, "level_lda_error", "level_lda_error_10_25", "level_lda_error_50"),
        combine(a, "level_qda_error", "level_qda_error_10_25", "level_qda_error_50"),
        combine(a, "meta_linear_fit", "meta_linear_fit"), combine(a, "meta_linear_coefficients", "meta_linear_coefficients"),
        combine(a, "meta_nonlinear_fit", "meta_interaction_fit", "meta_quadratic_fit"), combine(a, "meta_conditioning", "meta_conditioning"),
        combine(a, "ic_entropy", "ic_entropy"), combine(a, "ic_epsilon", "ic_epsilon"),
        combine(a, "nbc_fitness", "nbc_fitness"), combine(a, "nbc_neighbor_ratios", "nbc_neighbor_ratios"),
        combine(a, "pca_global_variance", "pca_global_variance"),
    ])
    p10 = dict([
        combine(a, "dispersion", "disp_difference", "disp_ratio"),
        combine(a, "objective_distribution", "distribution_shape", "distribution_peaks"),
        combine(a, "level_discriminability", "level_discriminability_10", "level_discriminability_25_50"),
        combine(a, "level_lda_error", "level_lda_error_10_25", "level_lda_error_50"),
        combine(a, "level_qda_error", "level_qda_error_10_25", "level_qda_error_50"),
        combine(a, "meta_linear", "meta_linear_fit", "meta_linear_coefficients"),
        combine(a, "meta_nonlinearity", "meta_interaction_fit", "meta_quadratic_fit", "meta_conditioning"),
        combine(a, "information_content", "ic_entropy", "ic_epsilon"),
        combine(a, "nearest_better_geometry", "nbc_fitness", "nbc_neighbor_ratios"),
        combine(a, "pca_variance_structure", "pca_global_variance"),
    ])
    p5 = dict([
        combine(a, "rank_and_value_concentration", "disp_difference", "disp_ratio", "distribution_shape", "distribution_peaks"),
        combine(a, "level_set_separability", "level_discriminability_10", "level_discriminability_25_50", "level_lda_error_10_25", "level_lda_error_50", "level_qda_error_10_25", "level_qda_error_50"),
        combine(a, "global_model_structure", "meta_linear_fit", "meta_linear_coefficients", "meta_interaction_fit", "meta_quadratic_fit", "meta_conditioning"),
        combine(a, "ruggedness_and_information", "ic_entropy", "ic_epsilon"),
        combine(a, "local_and_global_geometry", "nbc_fitness", "nbc_neighbor_ratios", "pca_global_variance"),
    ])
    return {"5": p5, "10": p10, "15": p15, "20": p20}


PANELS = {
    "BBOB_FILTERED": {
        "candidate_pool_size": 30,
        "source": "data/Ablation_ELA/Processed_ELA_Pipeline/pipeline_aligned_ela.csv",
        "preprocessing": "Per-problem Pearson correlation filtering had already been applied upstream; this panel uses the 30-column aligned usable pool.",
        "partitions": bbob_partitions(),
    },
    "MABBOB_LLM_COMMON_UNFILTERED": {
        "candidate_pool_size": 52,
        "sources": ["data/MABBOB/mabbob_selected_ela.csv", "data/LLM/llm_generated_ela.csv"],
        "preprocessing": "No correlation filter was found. The common 52-feature usable pool is used; LLM-only pca.expl_var.cov_init is excluded for identical MA-BBOB/LLM groups.",
        "partitions": common_partitions(),
    },
}


def validate() -> None:
    for panel, spec in PANELS.items():
        reference = None
        previous_groups = None
        for expected, groups in spec["partitions"].items():
            assert len(groups) == int(expected), (panel, expected, len(groups))
            flattened = [f for features in groups.values() for f in features]
            assert all(groups.values()), (panel, expected, "empty group")
            assert len(flattened) == len(set(flattened)), (panel, expected, "duplicate feature")
            assert len(flattened) == spec["candidate_pool_size"], (panel, expected, len(flattened))
            if reference is None:
                reference = set(flattened)
            assert set(flattened) == reference, (panel, expected, "pool mismatch")
            if previous_groups is not None:
                coarse_sets = [set(features) for features in previous_groups.values()]
                for name, features in groups.items():
                    assert any(set(features) <= coarse for coarse in coarse_sets), (
                        panel, expected, name, "not a hierarchical refinement"
                    )
            previous_groups = groups


def main() -> None:
    validate()
    payload = {
        "purpose": "Exhaustive hierarchical partitions for group-ablation studies; 5, 10, 15, and 20 denote numbers of groups, not numbers of selected features.",
        "design": "Groups follow ELA mechanisms. Existing family ablations and five-fold permutation importance determine which broad families are split at finer resolutions.",
        "empirical_basis": {
            "family_importance": "Five-fold group permutation importance ranks Dispersion highest on average, followed by Meta-model, Level-set, Nearest Better, Information Content, PCA, and Distribution.",
            "leave_family_out": "Existing RF ablations show source-dependent effects: NBC is strongest for BBOB; Level-set and Meta-model are strongest for MA-BBOB; effects are smaller and less stable for LLM. MLP ablations are scale-unstable for LLM/MA-BBOB, so they are supporting rather than primary evidence.",
            "individual_importance": "High mean feature importance is concentrated in nbc.nn_nb.mean_ratio, nbc.nb_fitness.cor, ic.m0, pca.expl_var.cor_init, nbc.nn_nb.sd_ratio, ela_meta.lin_simple.intercept, and disp.ratio_mean_02.",
        },
        "panels": PANELS,
    }
    (OUT / "feature_sets.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (OUT / "feature_sets_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "n_groups", "group_id", "group_name", "group_size", "feature"])
        for panel, spec in PANELS.items():
            for n_groups, groups in spec["partitions"].items():
                for group_id, (name, features) in enumerate(groups.items(), start=1):
                    for feature in features:
                        writer.writerow([panel, n_groups, group_id, name, len(features), feature])


if __name__ == "__main__":
    main()
