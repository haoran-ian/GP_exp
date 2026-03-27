import numpy as np

class HybridPSODE:
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
        self.F = 0.5 + np.random.rand() * 0.1  # Introduce adaptive DE mutation factor
        self.CR = 0.9 + np.random.rand() * 0.1  # Introduce adaptive DE crossover rate

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

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive = r1 * (self.personal_best_positions - self.particles)
            social = r2 * (self.global_best_position - self.particles)
            inertia_weight = 0.4 + np.random.sin(np.random.rand() * np.pi) / 2.5  # Introduce chaos mechanism
            self.velocities = inertia_weight * self.velocities + cognitive + social
            self.particles += self.velocities

            self.particles = np.clip(self.particles, lb, ub)

            if evaluations % (self.budget // 5) == 0:
                for i in range(self.population_size):
                    indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                    x1, x2, x3 = self.particles[indices]
                    mutant_vector = np.clip(x1 + self.F * (x2 - x3), lb, ub)
                    crossover = np.random.rand(self.dim) < self.CR
                    trial_vector = np.where(crossover, mutant_vector, self.particles[i])
                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector

        return self.global_best_position, self.global_best_score