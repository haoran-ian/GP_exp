import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.particles = self.chaotic_initialization()
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.F = 0.5
        self.CR = 0.9

    def chaotic_initialization(self):
        # Using a simple logistic map for chaotic sequence generation
        x = np.random.rand(self.population_size, self.dim)
        for _ in range(100):  # Chaotic sequence iterations
            x = 4 * x * (1 - x)  # Logistic map
        return x

    def update_parameters(self, evaluations):
        # Adaptive adjustment of parameters based on current budget usage
        progress = evaluations / self.budget
        self.F = 0.5 + 0.3 * progress
        self.CR = 0.9 - 0.4 * progress

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
            inertia_weight = 0.5 + np.random.rand() / 3  # Adjusted inertia weight
            self.velocities = inertia_weight * self.velocities + cognitive + social
            self.particles = np.clip(self.particles + self.velocities, lb, ub)

            if evaluations % (self.budget // 5) == 0:
                self.update_parameters(evaluations)
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