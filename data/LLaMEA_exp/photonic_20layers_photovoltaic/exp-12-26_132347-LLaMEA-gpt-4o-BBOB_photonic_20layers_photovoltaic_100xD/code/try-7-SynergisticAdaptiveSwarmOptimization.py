import numpy as np

class SynergisticAdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.w = 0.7  # Inertia weight
        self.population = None
        self.velocity = None
        self.best_agent = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.evaluations = 0
        self.entropy_threshold = 0.1  # Threshold for entropy-based adjustments

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.copy(fitness)
        self.evaluations += self.population_size

    def entropy_based_adaptation(self, fitness):
        prob = fitness / np.sum(fitness)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        if entropy < self.entropy_threshold:
            self.F *= 1.05
            self.CR *= 0.95
        else:
            self.F *= 0.95
            self.CR *= 1.05

    def differential_evolution_step(self, target_idx, func, fitness):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), lb, ub)

        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True

        trial = np.where(cross_points, mutant, self.population[target_idx])
        trial_fitness = func(trial)
        if trial_fitness < fitness[target_idx]:
            self.population[target_idx] = trial
            fitness[target_idx] = trial_fitness
            if trial_fitness < func(self.best_agent):
                self.best_agent = trial

    def particle_swarm_optimization_step(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocity[idx] = (self.w * self.velocity[idx] + 
                              self.c1 * r1 * (self.personal_best[idx] - self.population[idx]) +
                              self.c2 * r2 * (self.best_agent - self.population[idx]))
        self.population[idx] = np.clip(self.population[idx] + self.velocity[idx], func.bounds.lb, func.bounds.ub)

        current_fitness = func(self.population[idx])
        if current_fitness < self.personal_best_fitness[idx]:
            self.personal_best[idx] = self.population[idx]
            self.personal_best_fitness[idx] = current_fitness

    def __call__(self, func):
        self.initialize_population(func)
        fitness = np.array([func(ind) for ind in self.population])

        while self.evaluations < self.budget:
            self.entropy_based_adaptation(fitness)
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                self.differential_evolution_step(idx, func, fitness)
                self.evaluations += 1

                if self.evaluations < self.budget:
                    self.particle_swarm_optimization_step(idx, func)
                    self.evaluations += 1

        return self.best_agent