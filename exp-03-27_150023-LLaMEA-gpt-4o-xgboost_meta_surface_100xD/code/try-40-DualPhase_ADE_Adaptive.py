import numpy as np

class DualPhase_ADE_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.population = None
        self.fitness = None
        self.CR = 0.9  # Initial crossover probability
        self.F = 0.8   # Differential weight
        self.evaluations = 0
        self.memory = []
        self.memory_decay_rate = 0.95
        self.global_search_phase = True

    def init_population(self, bounds):
        lower_bound = bounds.lb
        upper_bound = bounds.ub
        self.population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def compute_diversity(self):
        mean_position = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - mean_position, axis=1))
        return diversity

    def adapt_parameters(self, recent_improvement):
        diversity = self.compute_diversity()
        if self.global_search_phase:
            self.CR = 0.6 + 0.3 * (diversity / np.sqrt(self.dim))
            self.F = 0.7 + 0.2 * (recent_improvement)
        else:
            self.CR = max(0.1, 0.5 * (diversity / np.sqrt(self.dim)))
            self.F = 0.3 + 0.5 * recent_improvement

    def switch_phase(self, improvement_rate):
        if improvement_rate < 1e-3:
            self.global_search_phase = False
        elif improvement_rate > 0.01:
            self.global_search_phase = True

    def differential_evolution(self, target_idx, bounds):
        idxs = [idx for idx in range(len(self.population)) if idx != target_idx]
        a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
        cross_points = np.random.rand(self.dim) < self.CR
        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def local_search(self, cand, bounds):
        intensity = 0.1 * (1 - self.evaluations / self.budget)
        perturbation = np.clip(cand + np.random.normal(0, intensity, self.dim), bounds.lb, bounds.ub)
        return perturbation

    def adaptive_memory_based_local_search(self, cand, bounds):
        if self.memory:
            recent_improvement = max(self.memory, key=lambda x: x[1]) if self.memory else (cand, np.inf)
            direction = recent_improvement[0] - cand
            intensity = 0.05 * (1 - self.evaluations / self.budget)
            perturbation = np.clip(cand + intensity * direction, bounds.lb, bounds.ub)
            return perturbation
        else:
            return self.local_search(cand, bounds)

    def decay_memory(self):
        self.memory = [(solution, fitness * self.memory_decay_rate) for solution, fitness in self.memory]
        self.memory = [(sol, fit) for sol, fit in self.memory if fit < np.inf]

    def evaluate_population(self, func):
        improvements = []
        for i in range(self.population_size):
            fitness_before = self.fitness[i]
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            improvements.append(abs(fitness_before - self.fitness[i]))
            if self.evaluations >= self.budget:
                return improvements
        return improvements

    def __call__(self, func):
        self.init_population(func.bounds)
        improvements = self.evaluate_population(func)

        while self.evaluations < self.budget:
            recent_improvement = np.mean(improvements)
            self.adapt_parameters(recent_improvement)
            self.switch_phase(recent_improvement)
            self.decay_memory()

            for i in range(len(self.population)):
                trial = self.differential_evolution(i, func.bounds)
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.memory.append((trial, trial_fitness))

                if np.random.rand() < 0.1:
                    local_candidate = self.adaptive_memory_based_local_search(self.population[i], func.bounds)
                    local_fitness = func(local_candidate)
                    self.evaluations += 1
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_candidate
                        self.fitness[i] = local_fitness
                        self.memory.append((local_candidate, local_fitness))

                if self.evaluations >= self.budget:
                    break

            improvements = self.evaluate_population(func)

        return self.population[np.argmin(self.fitness)]