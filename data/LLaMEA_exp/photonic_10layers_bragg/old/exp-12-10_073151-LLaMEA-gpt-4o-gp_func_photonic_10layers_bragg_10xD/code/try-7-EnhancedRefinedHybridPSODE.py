import numpy as np

class EnhancedRefinedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.9  # Adaptive inertia weight, starting high
        self.w_decay = 0.99  # Decay rate for inertia weight
        self.F = 0.8
        self.CR = 0.9
        self.population = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.adaptive_mutation_rate = 0.1  # Initial mutation rate for enhanced exploration

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, float('inf'))

    def update_personal_best(self, func):
        scores = np.apply_along_axis(func, 1, self.population)
        for i, score in enumerate(scores):
            if score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.population[i]
        return scores

    def update_global_best(self, scores):
        min_index = np.argmin(scores)
        if scores[min_index] < self.global_best_score:
            self.global_best_score = scores[min_index]
            self.global_best_position = self.population[min_index]

    def pso_step(self, func):
        scores = self.update_personal_best(func)
        self.update_global_best(scores)
        
        for i in range(self.population_size):
            r1, r2 = np.random.rand(2)
            cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
            social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
            self.population[i] += self.velocities[i]
        
        self.w *= self.w_decay  # Decaying inertia weight for balance between exploration and exploitation

    def de_step(self, bounds, func):
        lb, ub = bounds.lb, bounds.ub
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), lb, ub)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            if func(trial) < func(self.population[i]):
                new_population[i] = trial
            else:
                # Implement adaptive mutation if no improvement
                mutation_rate = self.adaptive_mutation_rate * np.random.rand(self.dim)
                mutated = np.clip(self.population[i] + mutation_rate * (ub - lb), lb, ub)
                new_population[i] = mutated if func(mutated) < func(self.population[i]) else self.population[i]
        self.population = new_population

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        evaluations = 0

        while evaluations < self.budget:
            if evaluations % 2 == 0:
                self.pso_step(func)
            else:
                self.de_step(bounds, func)
            evaluations += self.population_size

        return self.global_best_position, self.global_best_score