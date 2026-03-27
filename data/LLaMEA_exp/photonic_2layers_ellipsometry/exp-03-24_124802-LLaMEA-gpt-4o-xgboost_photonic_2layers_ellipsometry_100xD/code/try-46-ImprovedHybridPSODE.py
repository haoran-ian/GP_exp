import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.particles = np.random.rand(self.initial_population_size, self.dim)
        self.velocities = np.random.rand(self.initial_population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.F = 0.6
        self.CR = 0.95
        self.c1_max, self.c1_min = 2.5, 0.5
        self.c2_max, self.c2_min = 2.5, 0.5
        self.population_size = self.initial_population_size

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        self.particles = lb + (ub - lb) * self.particles
        evaluations = 0

        while evaluations < self.budget:
            # Adjust population size dynamically
            if evaluations % (self.budget // 10) == 0 and self.population_size > 20:
                self.population_size -= 5
            
            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            c1 = self.c1_max - (self.c1_max - self.c1_min) * (evaluations / self.budget)
            c2 = self.c2_min + (self.c2_max - self.c2_min) * (evaluations / self.budget)
            cognitive = c1 * r1 * (self.personal_best_positions[:self.population_size] - self.particles[:self.population_size])
            social = c2 * r2 * (self.global_best_position - self.particles[:self.population_size])
            inertia_weight = 0.4 + np.random.rand() / 2.5
            self.velocities[:self.population_size] = inertia_weight * self.velocities[:self.population_size] + cognitive + social
            self.particles[:self.population_size] += self.velocities[:self.population_size]
            self.particles[:self.population_size] = np.clip(self.particles[:self.population_size], lb, ub)

            if evaluations % (self.budget // 5) == 0:
                for i in range(self.population_size):
                    indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                    x1, x2, x3 = self.particles[indices]
                    dynamic_F = 0.5 + 0.3 * (1 - evaluations / self.budget)  # Dynamic mutation factor
                    mutant_vector = np.clip(x1 + dynamic_F * (x2 - x3), lb, ub)
                    crossover = np.random.rand(self.dim) < self.CR
                    trial_vector = np.where(crossover, mutant_vector, self.particles[i])
                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector

        return self.global_best_position, self.global_best_score