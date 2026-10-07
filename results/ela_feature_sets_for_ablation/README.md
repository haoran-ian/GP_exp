# ELA feature groups for ablation studies

This directory provides exhaustive partitions of the available ELA features into **5, 10, 15, and 20 groups**. The numbers refer to the number of groups, not the number of selected features. Every partition contains the complete feature pool, and each feature belongs to exactly one group.

Two panels are necessary because the upstream preprocessing differs:

| Panel | Feature pool | Use |
|---|---:|---|
| `BBOB_FILTERED` | 30 | BBOB data after the existing per-problem Pearson correlation filtering and cross-problem alignment |
| `MABBOB_LLM_COMMON_UNFILTERED` | 52 | The common usable MA-BBOB/LLM pool; no upstream correlation filter was found |

LLM has one additional usable feature, `pca.expl_var.cov_init`. It is excluded so that MA-BBOB and LLM use identical groups. This makes their ablation results directly comparable.

## Grouping logic

The partitions form a hierarchy. A group at a coarse resolution is split into mechanism-specific subgroups at finer resolutions. No feature moves across an unrelated mechanism. The five broad groups are:

1. Rank and value concentration: Dispersion and objective-value Distribution.
2. Level-set separability: LDA/QDA discriminability and classification errors.
3. Global model structure: linear, interaction, and quadratic meta-model behavior.
4. Ruggedness and information: Information Content entropy and epsilon measures.
5. Local and global geometry: Nearest-Better Clustering and PCA variance structure.

The 10-group partition restores the main ELA distinctions. The 15- and 20-group partitions split multi-scale or internally heterogeneous families by statistic, classifier, threshold, or geometry type. BBOB uses a slightly different fine hierarchy because correlation filtering removed many Dispersion, Information Content, NBC, and PCA variants.

## How the previous results affected the design

The existing experiments were used as evidence for the *resolution* of the groups, while ELA principles define their boundaries.

- Five-fold permutation importance ranks the broad families approximately as Dispersion, Meta-model, Level-set, Nearest Better, Information Content, PCA, and Distribution.
- RF leave-one-family-out ablations are source dependent: NBC has the largest effect for BBOB, while Level-set and Meta-model have the largest effects for MA-BBOB. LLM effects are smaller and less stable.
- The strongest individual mean importances include `nbc.nn_nb.mean_ratio`, `nbc.nb_fitness.cor`, `ic.m0`, `pca.expl_var.cor_init`, `nbc.nn_nb.sd_ratio`, `ela_meta.lin_simple.intercept`, and `disp.ratio_mean_02`.
- MLP ablations for LLM and MA-BBOB contain very large scale shifts, especially for PCA and the full set. They are treated as supporting evidence rather than the main grouping criterion.

These findings justify giving Dispersion, Meta-model, Level-set, NBC, IC, and PCA separate subgroups at finer resolutions. Correlated threshold variants remain together until the 20-group resolution, where appropriate.

## Recommended ablation protocol

For a partition with `K` groups, train one baseline with all features and `K` ablated models. Each ablated model removes exactly one group. Fit any scaler, imputer, or additional correlation filter inside each training fold. Report both predictive-performance change and algorithm-selection change, because the existing results show that these can disagree.

Use the same folds for every resolution. The 5-group partition gives the clearest high-level interpretation; 10 groups is the recommended primary analysis; 15 and 20 groups localize an observed effect.

## Files

- `feature_sets.json`: machine-readable panel definitions and all four partitions.
- `feature_sets_long.csv`: one row per panel, resolution, group, and feature.
- `build_feature_groupings.py`: reproducible generator with validation for group count, uniqueness, and complete coverage.

## Evidence files

- `results/Combined/feature_set_ablation_simple/overall_feature_set_ablation.csv`
- `results/Combined/source_specific_rf_realworld_ablation/overall_all_models_feature_set_ablation.csv`
- `results/Combined/source_specific_mlp_realworld_ablation/overall_all_models_feature_set_ablation.csv`
- `data/Combined/autogluon_feature_selection_then_retrain/feature_group_importance_all_folds.csv`
- `data/Combined/autogluon_feature_selection_then_retrain/feature_importance_all_folds.csv`
- `data/Combined/autogluon_feature_selection_then_retrain/feature_subset_leaderboard.csv`
