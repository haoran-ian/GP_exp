**Principle-based classification of 41 optimizers and selection of 8 representatives**

The selected algorithms are `ng_MultiBFGS`, `ng_Powell`, `ng_OnePlusOne`, `modcma_base`, `modde_base`, `modde_lshade`, `opy_GA`, and `opy_SAVPSO`. The selection aims to cover different search mechanisms. It was not derived from the observed AUC ranking and does not imply uniform superiority across problems.

The review used the four benchmark scripts' `TOP_ALGORITHMS` lists, their invocation parameters, references and update rules in the installed libraries, and accessible papers and official documentation. Review date: 2026-09-24. `modde_base` was subsequently included at the user's request; `opy_ABC`, `opy_CS`, and `opy_EO` were later removed from the recommended subset.

The inspected packages are in `/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages`: Opytimizer 3.1.2, Mealpy 2.5.3, Nevergrad 1.0.0, ModCMA 1.0.2, and ModDE 0.0.1. This is the currently accessible local environment; it has not been established that every historical experiment used these versions. The project's `requirements.txt` specifies a different ModCMA version and does not pin all the other packages. Implementation-specific observations below apply to the inspected versions.

All 25 Opytimizer classes contain references; GA cites a textbook. All seven Mealpy classes provide references or links, but `LevyES` and `BaseVCS` are explicitly library-developed variants. Nevergrad wrappers do not all have unique dedicated papers. The inventory distinguishes explicit library citations from supplementary theoretical references. For paywalled sources, the review used accessible bibliographic information, abstracts, and library code; it does not claim access to every full paper.

The classification is an analysis of search operators, not an official library taxonomy. Selection criteria include numerical gradient estimation, full covariance learning, population differences, recombination, velocity and historical memory, heavy-tailed steps, allocation of local search effort, and elite pools with time-dependent schedules. Biological or physical metaphors alone do not establish independent mechanisms.

The following table contains the eight selected algorithms. References support algorithm provenance; selection rationales are analytical judgments.

| Benchmark identifier | Core mechanism | Selection rationale | Reference and implementation |
|---|---|---|---|
| `ng_MultiBFGS` | Calls SciPy BFGS, estimates gradients by finite differences, and uses quasi-Newton updates; the underlying wrapper allows random restarts. | represents local slope and curvature information; this configuration creates one BFGS instance. | [reference](https://doi.org/10.1093/comjnl/13.3.317); [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/nevergrad/optimization/optimizerlib.py:2019) |
| `ng_Powell` | Performs one-dimensional searches along a set of directions and updates that set without constructing numerical gradients. | represents derivative-free direction-set local search. | [reference](https://doi.org/10.1093/comjnl/7.2.155); [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/nevergrad/optimization/recastlib.py:505) |
| `ng_OnePlusOne` | Generates one Gaussian-mutated offspring from one parent, retains the better solution, and adapts the global step size using success information. | a simple stochastic local-search baseline without covariance learning. | [reference](https://link.springer.com/article/10.1023/A:1015059928466); [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/nevergrad/optimization/optimizerlib.py:78) |
| `modcma_base` | Samples a multivariate Gaussian and adapts its mean, full covariance matrix, and step size to learn variable dependencies. | represents search-distribution adaptation and learning of variable interactions. | [reference](https://arxiv.org/abs/1604.00772); [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/modcma/modularcmaes.py:1) |
| `modde_base` | Fixed-parameter DE/rand/1/bin uses random population members and difference vectors, followed by crossover and greedy selection. | provides a basic-DE comparison against adaptive DE. | [reference](https://groups.csail.mit.edu/EVO-DesignOpt/pb/uploads/Site/differentialEv.pdf); [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/modde/parameters.py:120) |
| `modde_lshade` | Combines current-to-pbest differential mutation, an external archive, success-history adaptation of F/CR, and population reduction based on consumed evaluations. | represents adaptive DE; report it as this benchmark's modular L-SHADE configuration. | [reference](https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf); [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/modde/parameters.py:120) |
| `opy_GA` | Combines fitness-based selection, real-valued arithmetic crossover, Gaussian mutation, and survivor selection from parents and offspring. | represents recombination-centered evolutionary search; this is a real-coded GA. | [reference](https://doi.org/10.7551/mitpress/3927.001.0001); [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/evolutionary/ga.py:23) |
| `opy_SAVPSO` | Maintains velocity and personal-best memory and combines global-best guidance with differences between historical positions. | represents velocity-based search with memory; it is a PSO variant, not the complete constrained framework of its source paper. | [reference](https://link.springer.com/article/10.1007/s10898-007-9255-9); [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/pso.py:428) |

Keeping both `modcma_base` and `ng_OnePlusOne` preserves a meaningful distinction: full distribution-shape learning versus step-size adaptation around a single incumbent. Keeping both `ng_MultiBFGS` and `ng_Powell` contrasts finite-difference gradient information with direction-set line searches.

Among the seven DE-named algorithms, `modde_base` and `modde_lshade` provide a basic-versus-adaptive comparison. The former uses fixed-parameter DE/rand/1/bin. The latter combines current-to-pbest, success-history adaptation, an archive, and budget-dependent population reduction. It should retain its benchmark-specific configuration label.

The classification below covers all 41 algorithms.

| Mechanism family | Count | Algorithms |
|---|---:|---|
| Gaussian mutation and distribution adaptation | 5 | `modcma_bipop`, `modcma_base`, `ng_DiagonalCMA`, `ng_OnePlusOne`, `mealpy_LevyES` |
| Differential mutation and historical information | 9 | `modde_lshade`, `modde_base`, `ng_DE`, `opy_DE`, `opy_GCO`, `opy_BSA`, `mealpy_JADE`, `mealpy_SADE`, `mealpy_L_SHADE` |
| Numerical local search | 2 | `ng_MultiBFGS`, `ng_Powell` |
| Multiscale reproduction and random jumps | 3 | `opy_CS`, `opy_RRA`, `opy_IWO` |
| Interactions and time-dependent schedules | 7 | `opy_WEO`, `opy_LSA`, `opy_EO`, `opy_GOA`, `opy_TWO`, `opy_ASO`, `mealpy_OriginalTWO` |
| Role differentiation and population cooperation | 11 | `opy_QSA`, `opy_COA`, `opy_SOS`, `opy_AEO`, `opy_ABO`, `opy_SSA`, `opy_ABC`, `opy_RFO`, `opy_MVPA`, `mealpy_OriginalSARO`, `mealpy_BaseVCS` |
| Particle swarms with velocity memory | 2 | `opy_SAVPSO`, `opy_RPSO` |
| Selection and recombination | 1 | `opy_GA` |
| Arithmetic operator scheduling | 1 | `opy_AOA` |

These groups describe dominant operators and are not mathematically disjoint. GCO uses DE-like three-parent mutation, while BSA uses historical population differences, so both are grouped with differential methods. ASO also maintains velocity, but its forces derive from potentials. LevyES spans both evolution strategies and heavy-tailed jumps. Library directory names such as swarm, evolutionary, and science should not be treated as an independent taxonomy of search mechanisms.

Several implementation details affect interpretation and reporting:

1. **MultiBFGS depends on the worker configuration.** In the inspected source, the number of BFGS instances equals `num_workers`. The benchmark omits this argument, whose default is one. The underlying BFGS wrapper sets `random_restart=True`, so one instance can still restart. The local MultiBFGS docstring incorrectly mentions MetaCMA/SQP; the function body determines its classification. The call chain is `optimizerlib.MultiBFGS` to `recastlib.BFGS` to `scipy.optimize.minimize(method="BFGS")`, without a supplied `jac`. [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html) specifies forward-difference gradient estimation in this case. Describe it as a quasi-Newton method requiring no user-supplied gradients, rather than a method that uses no gradient information.

2. **The inspected Mealpy L_SHADE differs from the paper's population reduction.** In Mealpy 2.5.3, `L_SHADE.evolve` updates `dyn_pop_size` but still generates and selects candidates using `range(self.pop_size)` and does not truncate `self.pop` accordingly in that class. Reduction is scheduled by epoch, while the benchmark mainly terminates by `max_fe=100D`. Its minimum size is the initial `pop_size/5`, and archive insertion occurs after replacement. The source paper reduces the actual population using evaluation counts, sets a minimum of four, and archives displaced parents. These are static source observations, not results from a new performance experiment. They motivate preferring ModDE's configuration over `mealpy_L_SHADE` for this selection. [L-SHADE paper](https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf)

3. **ModDE also requires a precise configuration label.** The script sets `target/pbest`, `lpsr=True`, an archive, SHADE parameter adaptation, and an initial population of `18D`. The inspected implementation updates and physically truncates the population using `used_budget/budget`. However, the script does not explicitly set `memory_size=6`; the inspected default is 100. Boundary correction is `saturate`. Report this as the benchmark's modular L-SHADE configuration or variant, not an exact reproduction of the original authors' implementation. [ModDE repository](https://github.com/Dvermetten/ModDE)

4. **SAVPSO and ABC retain implementation-specific differences.** The SAVPSO paper includes DOCHM constraint handling, while the inspected class mainly implements velocity updates and boundary repair; this is insufficient to claim reproduction of the entire constrained method. Classical ABC scouts abandon stagnating food sources and generate random positions. The inspected `_send_scout` instead adds a uniform perturbation to an existing position and accepts an improvement. Use the label “Opytimizer ABC implementation.” Its role differentiation remains relevant, but the paper's complete procedure should not be assumed. [SAVPSO paper](https://link.springer.com/article/10.1007/s10898-007-9255-9); [ABC paper](https://link.springer.com/article/10.1007/s10898-007-9149-x)

5. **Two Mealpy variants lack verified dedicated papers.** LevyES explicitly changes the flow and equations and cites the Beyer-Schwefel ES review. BaseVCS changes the immune-response stage to a whole-position update and links the original VCS paper. Cite the underlying family reference together with the specific Mealpy variant and version.

6. **Disambiguate abbreviations and cross-library duplicates.** Here BSA means Backtracking Search; SSA means Salp Swarm; ABO means Artificial Butterfly Optimization; RPSO means Relativistic PSO; and AOA means Arithmetic Optimization. `opy_TWO` and `mealpy_OriginalTWO` share the Tug of War Optimization family. Two library implementations do not imply two independent theoretical families.

No new benchmark was run, and the selection was not adjusted using AUC. Before using the existing data for algorithm selection, check coverage, missing values, and actual evaluation budgets. The original Opytimizer wrapper approximates budgets through iteration counts, but evaluations per iteration can differ between methods. That comparability issue is separate from mechanism-based selection.

The following inventory provides the mechanism, selection decision, and provenance for every algorithm. Where no external paper link is supplied, the bibliography comes from the installed class documentation, which is linked for inspection. This does not imply independent full-text verification of every paper. The CSV also records package versions, source paths, line numbers, and SHA-256 hashes.

**1. `modcma_bipop` — Gaussian mutation and distribution adaptation**

Mechanism: Full-covariance CMA-ES with restart regimes using small and large populations. Decision: Shares the CMA-ES core; consider replacing the base variant when studying restarts or larger budgets.

Provenance: Shares the ModCMA framework references; a BIPOP restart reference is supplied separately. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/modcma/modularcmaes.py:1) (modcma 1.0.2).

Reference: Hansen (2009), Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed; Framework references are the same as for modcma_base. [paper/official resource](https://cma-es.github.io/).

**2. `modcma_base` — Gaussian mutation and distribution adaptation**

Mechanism: Samples a multivariate Gaussian and adapts its mean, full covariance matrix, and step size to learn variable dependencies. Decision: Selected: represents search-distribution adaptation and learning of variable interactions.

Provenance: Package METADATA/README explicitly cites the modular framework papers; CMA theory references are supplied separately. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/modcma/modularcmaes.py:1) (modcma 1.0.2).

Reference: van Rijn et al. (2016), Evolving the Structure of Evolution Strategies; de Nobel et al. (2021), Tuning as a Means of Assessing the Benefits of New Ideas in Interplay with Existing Algorithmic Modules. Theory: Hansen and Ostermeier (2001), Completely Derandomized Self-Adaptation in Evolution Strategies; Hansen (2016), The CMA Evolution Strategy: A Tutorial. [paper/official resource](https://arxiv.org/abs/1604.00772).

**3. `modde_lshade` — Differential mutation and historical information**

Mechanism: Combines current-to-pbest differential mutation, an external archive, success-history adaptation of F/CR, and population reduction based on consumed evaluations. Decision: Selected: represents adaptive DE; report it as this benchmark's modular L-SHADE configuration.

Provenance: Package README explicitly provides an L-SHADE configuration; the corresponding paper is supplied separately. Current parameters differ from the paper. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/modde/parameters.py:120) (modde 0.0.1).

Reference: Tanabe and Fukunaga (2014), Improving the Search Performance of SHADE Using Linear Population Size Reduction. [paper/official resource](https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf).

**4. `modde_base` — Differential mutation and historical information**

Mechanism: Fixed-parameter DE/rand/1/bin uses random population members and difference vectors, followed by crossover and greedy selection. Decision: Selected: provides a basic-DE comparison against adaptive DE.

Provenance: Package README describes modular DE; no dedicated paper for this configuration was found. The reference supplies family-level background. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/modde/parameters.py:120) (modde 0.0.1).

Reference: Storn and Price (1997), Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. [paper/official resource](https://groups.csail.mit.edu/EVO-DesignOpt/pb/uploads/Site/differentialEv.pdf).

**5. `ng_MultiBFGS` — Numerical local search**

Mechanism: Calls SciPy BFGS, estimates gradients by finite differences, and uses quasi-Newton updates; the underlying wrapper allows random restarts. Decision: Selected: represents local slope and curvature information; this configuration creates one BFGS instance.

Provenance: No dedicated MultiBFGS paper citation was found. The local call chain confirms BFGS; the theoretical reference is supplementary. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/nevergrad/optimization/optimizerlib.py:2019) (nevergrad 1.0.0).

Reference: Fletcher (1970), A new approach to variable metric algorithms. BFGS was independently developed by several authors; this is one foundational reference. See SciPy BFGS documentation for implementation details. [paper/official resource](https://doi.org/10.1093/comjnl/13.3.317).

**6. `ng_Powell` — Numerical local search**

Mechanism: Performs one-dimensional searches along a set of directions and updates that set without constructing numerical gradients. Decision: Selected: represents derivative-free direction-set local search.

Provenance: No dedicated paper citation for the Nevergrad Powell wrapper was found; the theoretical reference is supplementary. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/nevergrad/optimization/recastlib.py:505) (nevergrad 1.0.0).

Reference: Powell (1964), An efficient method for finding the minimum of a function of several variables without calculating derivatives. The implementation calls SciPy modified Powell. [paper/official resource](https://doi.org/10.1093/comjnl/7.2.155).

**7. `ng_DiagonalCMA` — Gaussian mutation and distribution adaptation**

Mechanism: Uses a diagonal covariance configuration of CMA to learn coordinate scales without full cross-variable correlations. Decision: Prioritize for a separability study; it is a structurally distinct variant within the CMA family.

Provenance: Constructed with diagonal=True in the library; the reference provides background on diagonal/separable CMA. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/nevergrad/optimization/optimizerlib.py:941) (nevergrad 1.0.0).

Reference: Ros and Hansen (2008), A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity. [paper/official resource](https://cma-es.github.io/).

**8. `ng_OnePlusOne` — Gaussian mutation and distribution adaptation**

Mechanism: Generates one Gaussian-mutated offspring from one parent, retains the better solution, and adapts the global step size using success information. Decision: Selected: a simple stochastic local-search baseline without covariance learning.

Provenance: Parameter documentation discusses Gaussian mutation and success-based step-size rules; there is no unique dedicated paper for this default wrapper. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/nevergrad/optimization/optimizerlib.py:78) (nevergrad 1.0.0).

Reference: Beyer and Schwefel (2002), Evolution strategies – A comprehensive introduction. This is a family-level review, not a dedicated paper for the default Nevergrad implementation. [paper/official resource](https://link.springer.com/article/10.1023/A:1015059928466).

**9. `ng_DE` — Differential mutation and historical information**

Mechanism: The inspected Nevergrad implementation forms a donor from the current individual, a random difference vector, and a direction toward the best solution, then applies crossover. Decision: Overlaps with the DE family; its name does not imply DE/rand/1.

Provenance: The class documentation describes DE settings. The basic-DE reference is supplementary, not an equation-level specification of this configuration. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/nevergrad/optimization/differentialevolution.py:307) (nevergrad 1.0.0).

Reference: Storn and Price (1997), Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. [paper/official resource](https://groups.csail.mit.edu/EVO-DesignOpt/pb/uploads/Site/differentialEv.pdf).

**10. `opy_CS` — Multiscale reproduction and random jumps**

Mechanism: Uses heavy-tailed Levy steps to generate candidates, together with nest replacement and local differential perturbations. Decision: Not selected in the final compact set.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/cs.py:21) (opytimizer 3.1.2).

Reference: X.-S. Yang and D. Suash. Cuckoo search via Lévy flights. World Congress on Nature & Biologically Inspired Computing (2009). [paper/official resource](https://arxiv.org/abs/1003.1594).

**11. `opy_DE` — Differential mutation and historical information**

Mechanism: Uses differential mutation from random individuals, crossover, and greedy replacement. Decision: Overlaps with basic ModDE; retaining both adds limited mechanism coverage to a compact set.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/evolutionary/de.py:21) (opytimizer 3.1.2).

Reference: R. Storn. On the usage of differential evolution for function optimization. Proceedings of North American Fuzzy Information Processing (1996).

**12. `opy_WEO` — Interactions and time-dependent schedules**

Mechanism: Uses evaporation-inspired phase scheduling, fitness-dependent evaporation masks, and population difference vectors. Decision: Combines differential updates and scheduling, overlapping with the selected DE and EO mechanisms.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/science/weo.py:20) (opytimizer 3.1.2).

Reference: A. Kaveh and T. Bakhshpoori. Water Evaporation Optimization: A novel physically inspired optimization algorithm. Computers & Structures (2016).

**13. `opy_QSA` — Role differentiation and population cooperation**

Mechanism: Allocates queues around leading solutions and applies staged Gamma-distributed perturbations and population differences. Decision: A candidate for grouped, multistage search; lower priority than the core mechanisms in this compact set.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/social/qsa.py:20) (opytimizer 3.1.2).

Reference: J. Zhang et al. Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems. Applied Mathematical Modelling (2018).

**14. `opy_COA` — Role differentiation and population cooperation**

Mechanism: Uses the best pack member and coordinate-wise pack medians for updates, with migration between packs. Decision: Represents subpopulations and population statistics; not selected in the final compact set.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/population/coa.py:20) (opytimizer 3.1.2).

Reference: J. Pierezan and L. Coelho. Coyote Optimization Algorithm: A New Metaheuristic for Global Optimization Problems. IEEE Congress on Evolutionary Computation (2018).

**15. `opy_SOS` — Role differentiation and population cooperation**

Mechanism: Uses mutualism, commensalism, and parasitism phases involving pairwise interactions and partial replacement. Decision: An optional twelfth algorithm to add explicit symbiotic pairwise interactions.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/sos.py:19) (opytimizer 3.1.2).

Reference: M.-Y. Cheng and D. Prayogo. Symbiotic Organisms Search: A new metaheuristic optimization algorithm. Computers & Structures (2014).

**16. `opy_AEO` — Role differentiation and population cooperation**

Mechanism: Combines production, consumption, and decomposition phases using bounded random sampling, population differences, and perturbations around the best solution. Decision: A multistage hybrid with operators overlapping those already selected.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/population/aeo.py:19) (opytimizer 3.1.2).

Reference: W. Zhao, L. Wang and Z. Zhang. Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm. Neural Computing and Applications (2019).

**17. `opy_ABO` — Role differentiation and population cooperation**

Mechanism: Separates sunspot and canopy individuals by fitness and uses neighbor-difference flight moves and fallback updates. Decision: Artificial Butterfly Optimization, distinct from BOA; overlaps with role-based and differential search.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/abo.py:20) (opytimizer 3.1.2).

Reference: X. Qi, Y. Zhu and H. Zhang. A new meta-heuristic butterfly-inspired algorithm. Journal of Computational Science (2017).

**18. `opy_GCO` — Differential mutation and historical information**

Mechanism: Uses survival counters to bias selection and generates candidates through three-parent differential mutation and crossover. Decision: Despite its immune-system terminology, its main candidate-generation operator resembles DE.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/population/gco.py:22) (opytimizer 3.1.2).

Reference: C. Villaseñor et al. Germinal center optimization algorithm. International Journal of Computational Intelligence Systems (2018).

**19. `opy_LSA` — Interactions and time-dependent schedules**

Mechanism: Combines lightning-inspired directions, exponential/Gaussian random steps, and an energy-decay schedule. Decision: Overlaps with multiscale random search and time-dependent scheduling.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/science/lsa.py:20) (opytimizer 3.1.2).

Reference: H. Shareef, A. Ibrahim and A. Mutlag. Lightning search algorithm. Applied Soft Computing (2015).

**20. `opy_RRA` — Multiscale reproduction and random jumps**

Mechanism: Combines large runner perturbations, small root searches, stagnation detection, and restarts. Decision: Provides multiscale local/global search; omitted to keep the set compact, with CS covering long jumps.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/evolutionary/rra.py:22) (opytimizer 3.1.2).

Reference: F. Merrikh-Bayat. The runner-root algorithm: A metaheuristic for solving unimodal and multimodal optimization problems inspired by runners and roots of plants in nature. Applied Soft Computing (2015).

**21. `opy_SAVPSO` — Particle swarms with velocity memory**

Mechanism: Maintains velocity and personal-best memory and combines global-best guidance with differences between historical positions. Decision: Selected: represents velocity-based search with memory; it is a PSO variant, not the complete constrained framework of its source paper.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/pso.py:428) (opytimizer 3.1.2).

Reference: H. Lu and W. Chen. Self-adaptive velocity particle swarm optimization for solving constrained optimization problems. Journal of global optimization (2008). [paper/official resource](https://link.springer.com/article/10.1007/s10898-007-9255-9).

**22. `opy_IWO` — Multiscale reproduction and random jumps**

Mechanism: Allocates offspring counts by fitness, reproduces through Gaussian dispersal with decreasing variance, and truncates the population competitively. Decision: Represents reproduction and variance scheduling; its botanical metaphor alone does not establish a separate search paradigm.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/evolutionary/iwo.py:19) (opytimizer 3.1.2).

Reference: A. R. Mehrabian and C. Lucas. A novel numerical optimization algorithm inspired from weed colonization. Ecological informatics (2006).

**23. `opy_EO` — Interactions and time-dependent schedules**

Mechanism: Maintains four elite solutions and their mean as an equilibrium pool and updates positions using time-dependent exponential and generation terms. Decision: Not selected in the final compact set.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/science/eo.py:20) (opytimizer 3.1.2).

Reference: A. Faramarzi et al. Equilibrium optimizer: A novel optimization algorithm. Knowledge-Based Systems (2020). [paper/official resource](https://www.sciencedirect.com/science/article/pii/S0950705119305295).

**24. `opy_SSA` — Role differentiation and population cooperation**

Mechanism: Moves a leader around the best solution while followers update through chain-like averaging. Decision: Salp Swarm Algorithm, not Sparrow Search Algorithm.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/ssa.py:16) (opytimizer 3.1.2).

Reference: S. Mirjalili et al. Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems. Advances in Engineering Software (2017).

**25. `opy_BSA` — Differential mutation and historical information**

Mechanism: Stores and shuffles a historical population, mutates using historical-current differences, and applies crossover and greedy selection. Decision: Backtracking Search, not Bird Swarm; an option for expanding the differential-search family.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/evolutionary/bsa.py:20) (opytimizer 3.1.2).

Reference: P. Civicioglu. Backtracking search optimization algorithm for numerical optimization problems. Applied Mathematics and Computation (2013).

**26. `opy_ABC` — Role differentiation and population cooperation**

Mechanism: Uses employed-bee search, onlooker allocation of search effort, and scout handling of stagnating locations. Decision: Not selected in the final compact set.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/abc.py:21) (opytimizer 3.1.2).

Reference: D. Karaboga and B. Basturk. A powerful and efficient algorithm for numerical function optimization: Artificial bee colony (ABC) algorithm. Journal of Global Optimization (2007). [paper/official resource](https://link.springer.com/article/10.1007/s10898-007-9149-x).

**27. `opy_GOA` — Interactions and time-dependent schedules**

Mechanism: Computes distance-dependent attraction/repulsion, contracts the interaction scale, and uses best-solution guidance. Decision: Distinct from PSO's personal-best velocity mechanism; can replace EO to emphasize spatial pairwise interactions.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/goa.py:20) (opytimizer 3.1.2).

Reference: S. Saremi, S. Mirjalili and A. Lewis. Grasshopper Optimisation Algorithm: Theory and application. Advances in Engineering Software (2017).

**28. `opy_GA` — Selection and recombination**

Mechanism: Combines fitness-based selection, real-valued arithmetic crossover, Gaussian mutation, and survivor selection from parents and offspring. Decision: Selected: represents recombination-centered evolutionary search; this is a real-coded GA.

Provenance: The class docstring cites Mitchell's GA textbook (1998 in the library), not a dedicated implementation paper; the publisher link describes the 1996 edition. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/evolutionary/ga.py:23) (opytimizer 3.1.2).

Reference: M. Mitchell. An introduction to genetic algorithms. MIT Press (1998). [paper/official resource](https://doi.org/10.7551/mitpress/3927.001.0001).

**29. `opy_TWO` — Interactions and time-dependent schedules**

Mechanism: Converts fitness into weights and updates solutions using pulls from stronger individuals, displacement, and noise. Decision: Shares its theoretical family with mealpy_OriginalTWO; avoid counting both as independent mechanisms.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/science/two.py:21) (opytimizer 3.1.2).

Reference: A. Kaveh. Tug of War Optimization. Advances in Metaheuristic Algorithms for Optimal Design of Structures (2016).

**30. `opy_RPSO` — Particle swarms with velocity memory**

Mechanism: Adds relativistic velocity handling to PSO updates. Decision: Another PSO variant alongside SAVPSO; RPSO here means Relativistic PSO.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/swarm/pso.py:339) (opytimizer 3.1.2).

Reference: M. Roder, G. H. de Rosa, L. A. Passos, A. L. D. Rossi and J. P. Papa. Harnessing Particle Swarm Optimization Through Relativistic Velocity. IEEE Congress on Evolutionary Computation (2020).

**31. `opy_RFO` — Role differentiation and population cooperation**

Mechanism: Combines relocation toward the best solution, local noticing moves, and updates of weak individuals near an elite habitat. Decision: A composite position-perturbation method; the compact set prioritizes more directly interpretable mechanisms.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/population/rfo.py:21) (opytimizer 3.1.2).

Reference: D. Polap and M. Woźniak. Red fox optimization algorithm. Expert Systems with Applications (2021).

**32. `opy_ASO` — Interactions and time-dependent schedules**

Mechanism: Derives masses from fitness, computes acceleration from atomic potentials and best-solution constraints, and updates velocity and position. Decision: Has velocity state but potential-field driving forces; can replace EO when emphasizing force-based models.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/science/aso.py:19) (opytimizer 3.1.2).

Reference: W. Zhao, L. Wang and Z. Zhang. A novel atom search optimization for dispersion coefficient estimation in groundwater. Future Generation Computer Systems (2019).

**33. `opy_AOA` — Arithmetic operator scheduling**

Mechanism: Switches probabilistically between multiplication/division and addition/subtraction around the best solution according to time. Decision: Arithmetic Optimization, not Aquila; distinct arithmetic operators do not automatically require a slot in the compact set.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/misc/aoa.py:16) (opytimizer 3.1.2).

Reference: L. Abualigah et al. The Arithmetic Optimization Algorithm. Computer Methods in Applied Mechanics and Engineering (2021).

**34. `opy_MVPA` — Role differentiation and population cooperation**

Mechanism: Uses within-team best solutions, the global best, and inter-team competition to guide updates. Decision: Most Valuable Player Algorithm; belongs to grouped cooperation and competition.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/opytimizer/optimizers/social/mvpa.py:21) (opytimizer 3.1.2).

Reference: H. Bouchekara. Most Valuable Player Algorithm: a novel optimization algorithm inspired from sport. Operational Research (2017).

**35. `mealpy_JADE` — Differential mutation and historical information**

Mechanism: Combines current-to-pbest mutation, an external archive, and successful-sample updates of the F/CR distribution means. Decision: A clear adaptive-DE alternative if a modular L-SHADE configuration is undesirable.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/mealpy/evolutionary_based/DE.py:158) (mealpy 2.5.3).

Reference: [1] Zhang, J. and Sanderson, A.C., 2009. JADE: adaptive differential evolution with optional external archive. IEEE Transactions on evolutionary computation, 13(5), pp.945-958. [paper/official resource](https://doi.org/10.1109/TEVC.2009.2014613).

**36. `mealpy_SADE` — Differential mutation and historical information**

Mechanism: Learns selection probabilities for two DE mutation strategies and adapts crossover parameters. Decision: The Qin-Suganthan SaDE variant, not a generic label for all self-adaptive DE or jDE.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/mealpy/evolutionary_based/DE.py:306) (mealpy 2.5.3).

Reference: [1] Qin, A.K. and Suganthan, P.N., 2005, September. Self-adaptive differential evolution algorithm for numerical optimization. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1785-1791). IEEE. [paper/official resource](https://doi.org/10.1109/CEC.2005.1554904).

**37. `mealpy_L_SHADE` — Differential mutation and historical information**

Mechanism: Declares success-history adaptation and linear population reduction, but the inspected 2.5.3 code still generates offspring using fixed pop_size. Decision: Not the preferred L-SHADE representative: the inspected implementation differs from the paper's actual population reduction.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/mealpy/evolutionary_based/DE.py:593) (mealpy 2.5.3).

Reference: [1] Tanabe, R. and Fukunaga, A.S., 2014, July. Improving the search performance of SHADE using linear population size reduction. In 2014 IEEE congress on evolutionary computation (CEC) (pp. 1658-1665). IEEE. [paper/official resource](https://metahack.org/CEC2014-Tanabe-Fukunaga.pdf).

**38. `mealpy_OriginalSARO` — Role differentiation and population cooperation**

Mechanism: Stores search clues, uses social and individual differential-search phases, and resets after stagnation. Decision: Adds cooperative search with memory; omitted from the compact selection.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/mealpy/human_based/SARO.py:146) (mealpy 2.5.3).

Reference: [1] Shabani, A., Asgarian, B., Gharebaghi, S.A., Salido, M.A. and Giret, A., 2019. A new optimization algorithm based on search and rescue operations. Mathematical Problems in Engineering, 2019. [paper/official resource](https://doi.org/10.1155/2019/2482543).

**39. `mealpy_OriginalTWO` — Interactions and time-dependent schedules**

Mechanism: Uses fitness-based weights, team pulling forces, random displacement, and boundary handling. Decision: Same theoretical family as opy_TWO; implementations across libraries are not independent principles.

Provenance: The class docstring explicitly provides a reference or a source-paper link. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/mealpy/physics_based/TWO.py:12) (mealpy 2.5.3).

Reference: [1] Kaveh, A., 2017. Tug of war optimization. In Advances in metaheuristic algorithms for optimal design of structures (pp. 451-487). Springer, Cham. [paper/official resource](https://www.researchgate.net/publication/332088054_Tug_of_War_Optimization_Algorithm).

**40. `mealpy_LevyES` — Gaussian mutation and distribution adaptation**

Mechanism: Generates Gaussian ES offspring and additional Levy offspring, then truncates their union with the parents. Decision: A library-developed hybrid; the general ES reference is not a dedicated paper for this variant.

Provenance: The class docstring provides a reference or link and explicitly identifies a developed version; this does not establish a dedicated paper for the variant. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/mealpy/evolutionary_based/ES.py:105) (mealpy 2.5.3).

Reference: [1] Beyer, H.G. and Schwefel, H.P., 2002. Evolution strategies–a comprehensive introduction. Natural computing, 1(1), pp.3-52. [paper/official resource](https://www.cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html).

**41. `mealpy_BaseVCS` — Role differentiation and population cooperation**

Mechanism: Combines virus diffusion, Gaussian infection around an elite mean, and differential immune-response updates. Decision: A library-developed version, not a direct equation-by-equation reproduction of original Virus Colony Search.

Provenance: The class docstring provides a reference or link and explicitly identifies a developed version; this does not establish a dedicated paper for the variant. [local source](/data/hyin/conda_envs/benchmark/lib/python3.10/site-packages/mealpy/bio_based/VCS.py:11) (mealpy 2.5.3).

Reference: Li, Zhao, Weng and Han (2016), A novel nature-inspired algorithm for optimization: Virus colony search. The class documentation directly supplies the DOI. [paper/official resource](https://doi.org/10.1016/j.advengsoft.2015.11.004).

For ModCMA, also cite the framework papers by [de Nobel et al. (2021)](https://doi.org/10.1145/3449726.3463167) and [van Rijn et al. (2016)](https://ieeexplore.ieee.org/document/7850138), as specified by the repository. For CMA theory, [Hansen's tutorial](https://arxiv.org/abs/1604.00772) is useful. The GA docstring cites Mitchell's 1998 textbook edition, whereas the [publisher link](https://doi.org/10.7551/mitpress/3927.001.0001) describes the 1996 edition; use the edition actually consulted in a formal bibliography.

`recommended_8.json` contains the identifiers for filtering results by `algname`. `algorithm_inventory_41.csv` contains the complete classification and provenance index. `email_draft.txt` provides a short English summary for the supervisor. The benchmark scripts, library implementations, and raw results have not been modified.
