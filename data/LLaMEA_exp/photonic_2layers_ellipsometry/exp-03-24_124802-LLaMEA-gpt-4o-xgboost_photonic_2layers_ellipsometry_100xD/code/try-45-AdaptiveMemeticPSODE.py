import numpy as np

class AdaptiveMemeticPSODE:
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
        self.F = 0.6
        self.CR = 0.9
        self.c1_max, self.c1_min = 2.5, 0.5
        self.c2_max, self.c2_min = 2.5, 0.5
        self.local_search_prob = 0.1  # Probability of applying local search

    def local_search(self, position, func, lb, ub):
        step_size = (ub - lb) * 0.05
        candidate = position + step_size * np.random.uniform(-1, 1, self.dim)
        candidate = np.clip(candidate, lb, ub)
        candidate_score = func(candidate)
        if candidate_score < func(position):
            return candidate, candidate_score
        return position, func(position)

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
            c1 = self.c1_max - (self.c1_max - self.c1_min) * (evaluations / self.budget)
            c2 = self.c2_min + (self.c2_max - self.c2_min) * (evaluations / self.budget)
            cognitive = c1 * r1 * (self.personal_best_positions - self.particles)
            social = c2 * r2 * (self.global_best_position - self.particles)
            inertia_weight = 0.4 + np.random.rand() / 2.5
            self.velocities = inertia_weight * self.velocities + cognitive + social
            self.particles += self.velocities
            self.particles = np.clip(self.particles, lb, ub)

            if evaluations % (self.budget // 5) == 0:
                for i in range(self.population_size):
                    indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                    x1, x2, x3 = self.particles[indices]
                    F_individual = np.random.uniform(0.4, 0.9)
                    mutant_vector = np.clip(x1 + F_individual * (x2 - x3), lb, ub)
                    crossover = np.random.rand(self.dim) < self.CR
                    trial_vector = np.where(crossover, mutant_vector, self.particles[i])
                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector

            # Apply local search with a certain probability
            for i in range(self.population_size):
                if np.random.rand() < self.local_search_prob:
                    new_position, new_score = self.local_search(self.particles[i], func, lb, ub)
                    if new_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = new_score
                        self.personal_best_positions[i] = new_position
                        if new_score < self.global_best_score:
                            self.global_best_score = new_score
                            self.global_best_position = new_position

        return self.global_best_position, self.global_best_score