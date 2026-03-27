import numpy as np

class HybridGAPSODE:
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
        self.F = 0.6  # DE mutation factor
        self.CR = 0.95  # DE crossover rate
        self.ga_crossover_rate = 0.7  # Genetic Algorithm crossover rate
        self.ga_mutation_rate = 0.01  # Genetic Algorithm mutation rate

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
            inertia_weight = 0.4 + np.random.rand() / 2.5
            self.velocities = inertia_weight * self.velocities + cognitive + social
            self.particles += self.velocities

            self.particles = np.clip(self.particles, lb, ub)

            # Genetic Algorithm inspired crossover and mutation
            if evaluations % (self.budget // 5) == 0:
                for i in range(self.population_size):
                    if np.random.rand() < self.ga_crossover_rate:
                        parents = np.random.choice(self.population_size, 2, replace=False)
                        cross_point = np.random.randint(1, self.dim-1)
                        trial_vector = np.concatenate((self.particles[parents[0], :cross_point],
                                                       self.particles[parents[1], cross_point:]))
                        if np.random.rand() < self.ga_mutation_rate:
                            mutation_index = np.random.randint(self.dim)
                            trial_vector[mutation_index] = lb[mutation_index] + \
                                (ub[mutation_index] - lb[mutation_index]) * np.random.rand()
                        trial_vector = np.clip(trial_vector, lb, ub)
                    else:
                        trial_vector = self.particles[i]

                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial_vector

        return self.global_best_position, self.global_best_score