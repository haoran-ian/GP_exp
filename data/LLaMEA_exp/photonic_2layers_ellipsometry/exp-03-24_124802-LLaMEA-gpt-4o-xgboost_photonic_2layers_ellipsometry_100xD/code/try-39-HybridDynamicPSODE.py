import numpy as np

class HybridDynamicPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.F_base = 0.6  # Base DE mutation factor
        self.CR_base = 0.95  # Base DE crossover rate

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.particles = lb + (ub - lb) * self.particles
        evaluations = 0

        while evaluations < self.budget:
            for i, particle in enumerate(self.particles):
                score = func(particle)
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = particle
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle

            inertia_weight = self.dynamic_inertia(evaluations)
            self.update_velocities(inertia_weight)
            self.update_positions(lb, ub)

            if evaluations % (self.budget // 5) == 0:
                self.perform_de_crossover(func, evaluations, lb, ub)

        return self.global_best_position, self.global_best_score

    def dynamic_inertia(self, evaluations):
        # Dynamic inertia weight that decreases over time to improve exploration initially and exploitation later
        max_inertia = 0.9
        min_inertia = 0.4
        return max_inertia - (max_inertia - min_inertia) * (evaluations / self.budget)

    def update_velocities(self, inertia_weight):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive = r1 * (self.personal_best_positions - self.particles)
        social = r2 * (self.global_best_position - self.particles)
        self.velocities = inertia_weight * self.velocities + cognitive + social

    def update_positions(self, lb, ub):
        self.particles += self.velocities
        self.particles = np.clip(self.particles, lb, ub)

    def perform_de_crossover(self, func, evaluations, lb, ub):
        # Adjust DE parameters dynamically
        F = self.F_base + 0.2 * np.random.rand()
        CR = self.CR_base - 0.1 * np.random.rand()
        for i in range(self.population_size):
            indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
            x1, x2, x3 = self.particles[indices]
            mutant_vector = np.clip(x1 + F * (x2 - x3), lb, ub)
            crossover = np.random.rand(self.dim) < CR
            trial_vector = np.where(crossover, mutant_vector, self.particles[i])
            trial_score = func(trial_vector)
            evaluations += 1
            if trial_score < self.personal_best_scores[i]:
                self.personal_best_scores[i] = trial_score
                self.personal_best_positions[i] = trial_vector