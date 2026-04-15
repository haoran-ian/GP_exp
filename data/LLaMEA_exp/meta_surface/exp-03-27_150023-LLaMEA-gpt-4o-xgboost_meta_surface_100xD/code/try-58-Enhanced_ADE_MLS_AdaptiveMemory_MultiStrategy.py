import numpy as np

class Enhanced_ADE_MLS_AdaptiveMemory_MultiStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, 10 * dim)
        self.population = None
        self.fitness = None
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.evaluations = 0
        self.local_memory = []
        self.global_memory = []
        self.memory_decay_rate = 0.95
        self.dynamic_population_factor = 1.0

    def init_population(self, bounds):
        lower_bound = bounds.lb
        upper_bound = bounds.ub
        self.population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def compute_diversity(self):
        mean_position = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - mean_position, axis=1))
        return diversity

    def adapt_parameters(self):
        diversity = self.compute_diversity()
        diversity_factor = diversity / np.sqrt(self.dim)
        self.CR = 0.5 + 0.4 * diversity_factor

        local_improvement = max([fit for _, fit in self.local_memory], default=0)
        global_improvement = max([fit for _, fit in self.global_memory], default=0)

        if global_improvement > local_improvement:
            self.CR = max(0.1, min(self.CR * global_improvement / np.mean(self.fitness), 0.9))
            self.F = 0.5 + 0.3 * global_improvement
        else:
            self.CR = max(0.1, min(self.CR * local_improvement / np.mean(self.fitness), 0.9))
            self.F = 0.5 + 0.3 * local_improvement

    def adjust_population_size(self):
        if self.evaluations > self.budget // 2:
            self.dynamic_population_factor = 0.5 + 0.5 * (self.budget - self.evaluations) / (self.budget // 2)
            new_size = int(self.population_size * self.dynamic_population_factor)
            self.population = self.population[:new_size]
            self.fitness = self.fitness[:new_size]

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

    def memory_based_local_search(self, cand, bounds, is_global):
        if is_global and self.global_memory:
            recent_improvement = max(self.global_memory, key=lambda x: x[1])
            direction = recent_improvement[0] - cand
        elif self.local_memory:
            recent_improvement = max(self.local_memory, key=lambda x: x[1])
            direction = recent_improvement[0] - cand
        else:
            return self.local_search(cand, bounds)

        intensity = 0.05 * (1 - self.evaluations / self.budget)
        perturbation = np.clip(cand + intensity * direction, bounds.lb, bounds.ub)
        return perturbation

    def decay_memory(self):
        self.local_memory = [(solution, fitness * self.memory_decay_rate) for solution, fitness in self.local_memory]
        self.global_memory = [(solution, fitness * self.memory_decay_rate) for solution, fitness in self.global_memory]

    def __call__(self, func):
        self.init_population(func.bounds)
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.population[np.argmin(self.fitness)]

        while self.evaluations < self.budget:
            self.adapt_parameters()
            self.adjust_population_size()
            self.decay_memory()

            for i in range(len(self.population)):
                trial = self.differential_evolution(i, func.bounds)
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.local_memory.append((trial, trial_fitness))
                    if len(self.local_memory) > 5:
                        self.global_memory.append(self.local_memory.pop(0))

                if np.random.rand() < 0.1:
                    is_global_search = self.evaluations < self.budget // 3
                    local_candidate = self.memory_based_local_search(self.population[i], func.bounds, is_global_search)
                    local_fitness = func(local_candidate)
                    self.evaluations += 1
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_candidate
                        self.fitness[i] = local_fitness
                        self.local_memory.append((local_candidate, local_fitness))
                        if len(self.local_memory) > 5:
                            self.global_memory.append(self.local_memory.pop(0))

                if self.evaluations >= self.budget:
                    break

        return self.population[np.argmin(self.fitness)]