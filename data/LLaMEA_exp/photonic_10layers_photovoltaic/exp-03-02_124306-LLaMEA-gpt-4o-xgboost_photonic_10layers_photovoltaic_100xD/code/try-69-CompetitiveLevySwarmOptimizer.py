import numpy as np
from scipy.stats import levy_stable

class CompetitiveLevySwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.subpopulations = 5
        self.subpopulation_size = 10
        self.initial_population_size = self.subpopulations * self.subpopulation_size
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.F = 0.8
        self.CR = 0.9
        self.memory_size = 10
        self.memory = []
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.function_evaluations = 0

    def _initialize(self, bounds):
        n = self.initial_population_size * self.dim
        x = np.linspace(0, 1, n)
        logistic_map = 4 * x * (1 - x)
        chaotic_sequence = logistic_map.reshape(self.initial_population_size, self.dim)
        self.population = bounds.lb + (bounds.ub - bounds.lb) * chaotic_sequence
        self.velocities = np.random.uniform(-abs(bounds.ub - bounds.lb), abs(bounds.ub - bounds.lb), (self.initial_population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)

    def _evaluate(self, func):
        scores = np.array([func(p) for p in self.population])
        self.function_evaluations += self.initial_population_size
        for i in range(self.initial_population_size):
            if scores[i] < self.personal_best_scores[i]:
                self.personal_best_scores[i] = scores[i]
                self.personal_best_positions[i] = self.population[i]
            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.population[i]
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        self.memory.append((self.global_best_position, self.global_best_score))

    def _update_particles(self, bounds):
        for i in range(self.initial_population_size):
            r1, r2 = np.random.rand(2)
            self.w = 0.6 + 0.2 * np.random.rand()
            inertia = self.w * self.velocities[i]
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
            social = self.c2 * r2 * (self.global_best_position - self.population[i])
            self.velocities[i] = inertia + cognitive + social
            self.population[i] += self.velocities[i]
            self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)

    def _adaptive_mutation(self, func, bounds):
        dynamic_CR = self.CR * (1 + 0.05 * np.sin(self.function_evaluations / 30))
        for i in range(self.initial_population_size):
            if np.random.rand() < dynamic_CR:
                indices = np.random.choice(self.initial_population_size, 3, replace=False)
                x1, x2, x3 = self.population[indices]
                mutant = x1 + self.F * (x2 - x3)
                trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, self.population[i])
                trial = np.clip(trial, bounds.lb, bounds.ub)
                trial_score = func(trial)
                self.function_evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.population[i] = trial
                    self.personal_best_scores[i] = trial_score
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = trial

    def _levy_flight(self, bounds):
        alpha = 1.3
        for i in range(self.initial_population_size):
            step = levy_stable.rvs(alpha, 0, size=self.dim)
            step_size = 0.01 * step * (self.population[i] - self.global_best_position)
            noise = 0.05 * np.random.randn(self.dim)
            self.population[i] += step_size + noise
            self.population[i] = np.clip(self.population[i], bounds.lb, bounds.ub)

    def _maintain_diversity(self, bounds):
        diversity = np.std(self.population, axis=0).mean()
        if diversity < 0.15:
            self.population = bounds.lb + np.random.rand(self.initial_population_size, self.dim) * (bounds.ub - bounds.lb)

    def _competitive_co_evolution(self, bounds):
        subpopulations = [self.population[i*self.subpopulation_size:(i+1)*self.subpopulation_size] for i in range(self.subpopulations)]
        sub_scores = [np.sum(self.personal_best_scores[i*self.subpopulation_size:(i+1)*self.subpopulation_size]) for i in range(self.subpopulations)]
        best_subpopulation_index = np.argmin(sub_scores)
        for i in range(self.subpopulations):
            if i != best_subpopulation_index:
                improvement_factor = np.random.uniform(1.05, 1.2)
                subpop = subpopulations[i] * improvement_factor
                self.population[i*self.subpopulation_size:(i+1)*self.subpopulation_size] = np.clip(subpop, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        self._initialize(bounds)
        while self.function_evaluations < self.budget:
            self._evaluate(func)
            self._update_particles(bounds)
            if np.random.rand() < 0.6:
                self._adaptive_mutation(func, bounds)
            self._levy_flight(bounds)
            self._maintain_diversity(bounds)
            self._competitive_co_evolution(bounds)
        return self.global_best_position, self.global_best_score