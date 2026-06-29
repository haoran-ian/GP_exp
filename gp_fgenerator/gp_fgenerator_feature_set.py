
import os
import time
import random
import operator
import numpy as np
import pandas as pd
from .create_pset import create_pset
from .symb_regression import symb_regr
from .utils import logger2csv, fitness2df, result2csv
from .initialize_tree import genHalfAndHalf_
from deap import algorithms, base, creator, tools, gp
from scoop import futures
from functools import partial


def infer_ela_group(feature_name: str) -> str:
    """
    Infer a coarse ELA group from a feature name.

    This is intentionally permissive so that both names like
    'disp.ratio_mean_02' and 'dispersion.ratio_mean_02' map to 'disp'.
    """
    f = str(feature_name)

    if f.startswith("ela_meta") or "ela_meta" in f:
        return "ela_meta"
    if f.startswith("ela_distr") or f.startswith("ela_distribution") or "ela_distr" in f or "ela_distribution" in f:
        return "ela_distr"
    if f.startswith("ela_level") or "ela_level" in f:
        return "ela_level"
    if f.startswith("nbc") or ".nbc" in f or "nbc." in f:
        return "nbc"
    if f.startswith("ic") or f.startswith("information_content") or "information_content" in f or "ic." in f:
        return "ic"
    if f.startswith("disp") or f.startswith("dispersion") or "dispersion" in f or "disp." in f:
        return "disp"
    if f.startswith("pca") or "pca." in f:
        return "pca"
    if f.startswith("limo") or "limo." in f:
        return "limo"

    # fallback: use the first token before "."
    return f.split(".")[0]


def normalize_ela_group_name(group_name: str) -> str:
    """
    Normalize user-provided group aliases.

    Examples:
      'dispersion' -> 'disp'
      'distribution' -> 'ela_distr'
      'PCA' -> 'pca'
    """
    g = str(group_name).strip().lower()
    g = g.replace("-", "_").replace(" ", "_")

    aliases = {
        "meta": "ela_meta",
        "meta_model": "ela_meta",
        "metamodel": "ela_meta",
        "ela_meta": "ela_meta",

        "distribution": "ela_distr",
        "ela_distribution": "ela_distr",
        "ela_distr": "ela_distr",
        "distr": "ela_distr",

        "level": "ela_level",
        "level_set": "ela_level",
        "ela_level": "ela_level",

        "nearest_better": "nbc",
        "nearest_better_clustering": "nbc",
        "nbc": "nbc",

        "information_content": "ic",
        "info_content": "ic",
        "icofis": "ic",
        "ic": "ic",

        "dispersion": "disp",
        "disp": "disp",

        "pca": "pca",
        "principal_component_analysis": "pca",

        "linear_model": "limo",
        "limo": "limo",
    }
    return aliases.get(g, g)


def to_plain_dict(x):
    """
    Convert common vector containers to a plain dict when possible.
    """
    if x is None:
        return {}

    if isinstance(x, dict):
        return dict(x)

    if isinstance(x, pd.Series):
        return x.to_dict()

    # Keep unsupported formats unchanged elsewhere.
    return x


class GP_func_generator:
    def __init__(self,
                 doe_x,
                 target_vector,
                 minimization: bool = True,
                 bs_ratio: float = 0.8,
                 bs_repeat: int = 2,
                 list_ela: list = None,
                 ela_min: dict = None,
                 ela_max: dict = None,
                 ela_weight: dict = None,
                 dist_metric: str = 'cityblock',
                 problem_label: str = '',
                 filepath_save: str = '',
                 tree_size: tuple = (8, 12),
                 population: int = 100,
                 cxpb: float = 0.5,
                 mutpb: float = 0.1,
                 ngen: int = 10,
                 nhof: int = 1,
                 seed: int = 1,
                 verbose: bool = True,
                 # ------------------------------------------------------------
                 # New feature-set control
                 # ------------------------------------------------------------
                 use_ela_features: list = None,
                 exclude_ela_features: list = None,
                 use_ela_groups: list = None,
                 exclude_ela_groups: list = None,
                 feature_set_name: str = ''
                 ):
        """
        GP-based proxy function generator.

        New feature-set parameters
        --------------------------
        use_ela_features:
            Exact ELA feature names to keep. If None, start from all available
            features in target_vector.

        exclude_ela_features:
            Exact ELA feature names to remove.

        use_ela_groups:
            ELA groups to keep, e.g. ['ela_meta', 'nbc', 'ic'].

        exclude_ela_groups:
            ELA groups to remove, e.g. ['disp', 'pca'] or
            ['dispersion', 'PCA'].

        feature_set_name:
            Optional label saved to feature_set_used.csv for bookkeeping.

        Typical usage
        -------------
        Exclude dispersion and PCA:

            GP_func_generator(
                doe_x=X,
                target_vector=t,
                list_ela=list_ela,
                ela_min=ela_min,
                ela_max=ela_max,
                ela_weight=ela_weight,
                exclude_ela_groups=['dispersion', 'PCA'],
            )

        Use only an ablation-selected feature set:

            GP_func_generator(
                doe_x=X,
                target_vector=t,
                list_ela=list_ela,
                ela_min=ela_min,
                ela_max=ela_max,
                ela_weight=ela_weight,
                use_ela_features=top_features,
                feature_set_name='ABLATION_TOP_20',
            )
        """
        self.doe_x = doe_x
        self.minimization = minimization
        self.bs_ratio: float = bs_ratio
        self.bs_repeat: int = bs_repeat

        self.raw_target_vector = target_vector
        self.raw_list_ela: list = list_ela if list_ela is not None else []
        self.raw_ela_min: dict = ela_min if ela_min is not None else {}
        self.raw_ela_max: dict = ela_max if ela_max is not None else {}
        self.raw_ela_weight: dict = ela_weight if ela_weight is not None else {}

        self.use_ela_features = use_ela_features
        self.exclude_ela_features = exclude_ela_features
        self.use_ela_groups = use_ela_groups
        self.exclude_ela_groups = exclude_ela_groups
        self.feature_set_name = feature_set_name

        (
            self.target_vector,
            self.list_ela,
            self.ela_min,
            self.ela_max,
            self.ela_weight,
            self.selected_ela_features,
            self.selected_ela_groups,
        ) = self._prepare_ela_feature_set(
            target_vector=target_vector,
            list_ela=self.raw_list_ela,
            ela_min=self.raw_ela_min,
            ela_max=self.raw_ela_max,
            ela_weight=self.raw_ela_weight,
            use_ela_features=use_ela_features,
            exclude_ela_features=exclude_ela_features,
            use_ela_groups=use_ela_groups,
            exclude_ela_groups=exclude_ela_groups,
        )

        self.dist_metric: str = dist_metric
        self.problem_label: str = problem_label if problem_label else 'problem'
        self.filepath_save: str = filepath_save if filepath_save else os.path.join(
            os.getcwd(), f'results_gpfg_{self.doe_x.shape[1]}d_{self.problem_label}')
        self.tree_size: tuple = tree_size
        self.population: int = population
        self.cxpb: float = cxpb
        self.mutpb: float = mutpb
        self.ngen: int = ngen
        self.nhof: int = nhof
        self.seed: int = seed
        self.verbose: bool = verbose
        self.neval = 0
        self.weight = -1.0 if self.minimization else 1.0
        self.fopt = np.inf if self.minimization else -1.0*np.inf
        self.result = pd.DataFrame()
        self.id_best = np.inf
        self.err = {
            'syntax_err': 0,
            'y_err': 0,
            'ela_err': 0,
            'dist_err': 0,
            'success': 0,
        }

        if not os.path.isdir(self.filepath_save):
            os.makedirs(self.filepath_save)

        self._save_feature_set_metadata()

        if self.verbose:
            print("[GPFG] ELA feature-set configuration:")
            print(
                f"  feature_set_name: {self.feature_set_name if self.feature_set_name else '<unnamed>'}")
            print(
                f"  selected feature count: {len(self.selected_ela_features)}")
            print(f"  selected groups: {sorted(self.selected_ela_groups)}")

    def _prepare_ela_feature_set(self,
                                 target_vector,
                                 list_ela,
                                 ela_min,
                                 ela_max,
                                 ela_weight,
                                 use_ela_features=None,
                                 exclude_ela_features=None,
                                 use_ela_groups=None,
                                 exclude_ela_groups=None):
        """
        Filter all ELA-related inputs consistently.

        The original implementation passes target_vector, list_ela, ela_min,
        ela_max, and ela_weight directly to symb_regr. This function makes sure
        that if a feature set excludes, for example, dispersion and PCA, every
        one of those objects is filtered accordingly before symb_regr sees them.
        """
        target_dict = to_plain_dict(target_vector)
        ela_min = dict(ela_min) if ela_min is not None else {}
        ela_max = dict(ela_max) if ela_max is not None else {}
        ela_weight = dict(ela_weight) if ela_weight is not None else {}

        # Determine the universe of exact feature names.
        # Usually target_vector is a dict/Series keyed by exact ELA feature names.
        if isinstance(target_dict, dict):
            all_features = list(target_dict.keys())
        else:
            # Fallback: if target_vector is not keyed, use list_ela as the feature list.
            all_features = list(list_ela)

        all_features = [str(f) for f in all_features]

        selected = set(all_features)

        if use_ela_features is not None:
            selected &= set(str(f) for f in use_ela_features)

        if exclude_ela_features is not None:
            selected -= set(str(f) for f in exclude_ela_features)

        if use_ela_groups is not None:
            allowed_groups = set(normalize_ela_group_name(g)
                                 for g in use_ela_groups)
            selected = {f for f in selected if infer_ela_group(
                f) in allowed_groups}

        if exclude_ela_groups is not None:
            excluded_groups = set(normalize_ela_group_name(g)
                                  for g in exclude_ela_groups)
            selected = {f for f in selected if infer_ela_group(
                f) not in excluded_groups}

        selected_features = [f for f in all_features if f in selected]

        if len(selected_features) == 0:
            raise ValueError(
                "No ELA features remain after feature-set filtering. "
                "Check use_ela_features / exclude_ela_features / use_ela_groups / exclude_ela_groups."
            )

        selected_groups = set(infer_ela_group(f) for f in selected_features)

        # Filter target vector.
        if isinstance(target_dict, dict):
            filtered_target_vector = {
                f: target_dict[f] for f in selected_features if f in target_dict}
        else:
            # For non-dict target_vector, fall back to the original object.
            # Exact filtering is not safe without feature names.
            filtered_target_vector = target_vector

        # Filter normalization and weight dictionaries.
        filtered_ela_min = {f: ela_min[f]
                            for f in selected_features if f in ela_min}
        filtered_ela_max = {f: ela_max[f]
                            for f in selected_features if f in ela_max}
        filtered_ela_weight = {f: ela_weight[f]
                               for f in selected_features if f in ela_weight}

        # Filter list_ela.
        #
        # There are two common conventions:
        #   1. list_ela contains exact feature names.
        #   2. list_ela contains feature groups requested from the ELA calculator.
        #
        # We support both. If entries match exact features, keep exact selected
        # features. Otherwise, keep only selected groups.
        list_ela_as_str = [str(x) for x in list_ela]

        if any(x in all_features for x in list_ela_as_str):
            filtered_list_ela = [
                x for x in list_ela_as_str if x in selected_features]
        else:
            normalized_list_groups = [
                normalize_ela_group_name(x) for x in list_ela_as_str]
            filtered_list_ela = []
            for original, normalized in zip(list_ela_as_str, normalized_list_groups):
                if normalized in selected_groups:
                    filtered_list_ela.append(original)

            # If original list_ela was empty or incompatible, use selected group names.
            if not filtered_list_ela:
                filtered_list_ela = sorted(selected_groups)

        return (
            filtered_target_vector,
            filtered_list_ela,
            filtered_ela_min,
            filtered_ela_max,
            filtered_ela_weight,
            selected_features,
            selected_groups,
        )

    def _save_feature_set_metadata(self):
        rows = []
        for f in self.selected_ela_features:
            rows.append({
                "feature_set_name": self.feature_set_name if self.feature_set_name else "custom",
                "feature": f,
                "group": infer_ela_group(f),
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.filepath_save,
                  "feature_set_used.csv"), index=False)

        config = {
            "feature_set_name": self.feature_set_name,
            "n_selected_features": len(self.selected_ela_features),
            "selected_groups": sorted(self.selected_ela_groups),
            "use_ela_features": self.use_ela_features,
            "exclude_ela_features": self.exclude_ela_features,
            "use_ela_groups": self.use_ela_groups,
            "exclude_ela_groups": self.exclude_ela_groups,
            "list_ela_passed_to_symb_regr": self.list_ela,
        }
        pd.Series(config).to_json(
            os.path.join(self.filepath_save, "feature_set_config.json"),
            indent=2,
        )

    def evalSymbReg(self, individual, points):
        self.neval += 1
        start_time = time.time()
        func_ = self.toolbox.compile(expr=individual)

        fitness_ = symb_regr(
            func_,
            self.target_vector,
            self.bs_ratio,
            self.bs_repeat,
            self.list_ela,
            self.ela_min,
            self.ela_max,
            self.ela_weight,
            self.dist_metric,
            self.verbose,
            points,
        )

        self.err[fitness_[0]] += 1
        self.result = pd.concat(
            [self.result, fitness2df(
                str(individual), fitness_, label=f'{self.neval}')],
            axis=0,
            ignore_index=True,
        )

        if fitness_[1] < self.fopt:
            self.fopt = fitness_[1]
            self.id_best = self.neval

        if self.verbose:
            print(
                f'[GPFG] neval: {self.neval}, {fitness_[0]}, f: {fitness_[1]}; '
                f'fbest: {self.fopt}; time_cost: {time.time()-start_time:.2f}s; '
                f'id_best: {self.id_best}; n_ela_features: {len(self.selected_ela_features)}'
            )

        return fitness_[1],

    def __call__(self):
        if self.verbose:
            print(
                f'[GPFG] Started for {self.problem_label}; Pop: {self.population}; Gen: {self.ngen}.'
            )

        np.random.seed(self.seed)
        random.seed(self.seed + 10)

        pset = create_pset()
        genHalfAndHalf = partial(genHalfAndHalf_, self.doe_x)

        # Avoid DEAP creator duplicate-name errors when multiple GP_func_generator
        # instances are created in the same Python process.
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(self.weight,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree,
                           fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("map", futures.map)
        self.toolbox.register(
            "expr",
            genHalfAndHalf,
            pset=pset,
            min_=self.tree_size[0],
            max_=self.tree_size[1],
        )
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

        self.toolbox.register("evaluate", self.evalSymbReg, points=self.doe_x)
        self.toolbox.register("select", tools.selTournament, tournsize=5)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform,
                              expr=self.toolbox.expr_mut, pset=pset)

        self.toolbox.decorate("mate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate("mutate", gp.staticLimit(
            key=operator.attrgetter("height"), max_value=17))

        pop = self.toolbox.population(n=self.population)
        hof = tools.HallOfFame(self.nhof)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, logger = algorithms.eaSimple(
            pop,
            self.toolbox,
            self.cxpb,
            self.mutpb,
            self.ngen,
            stats=mstats,
            halloffame=hof,
            verbose=self.verbose,
        )

        result2csv(self.filepath_save, self.result, self.id_best)
        logger2csv(os.path.join(self.filepath_save, 'gpfg_logger.csv'), logger)

        if self.verbose:
            print(
                f'[GPFG] Finished: {self.err}; Success rate: {self.err["success"]/self.neval}.'
            )

        return hof, pop
# END CLASS
