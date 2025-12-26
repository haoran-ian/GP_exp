import numpy as np

class HierarchicalVariableNeighborhoodOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.particles = np.random.rand(self.population_size, dim)
        self.velocities = np.random.rand(self.population_size, dim) * 0.1
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.global_best = None
        self.global_best_score = float('inf')

    def adaptive_learning_rate(self, eval_count):
        return 0.5 + 0.5 * np.cos(np.pi * eval_count / self.budget)

    def variable_neighborhood_search(self, position, radius, lower_bound, upper_bound):
        perturbation = np.random.uniform(-radius, radius, size=position.shape)
        trial_position = position + perturbation
        return np.clip(trial_position, lower_bound, upper_bound)

    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        eval_count = 0

        while eval_count < self.budget:
            for i in range(self.population_size):
                score = func(self.particles[i])

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            adaptive_rate = self.adaptive_learning_rate(eval_count)
            inertia_weight = 0.9 - 0.5 * (eval_count / self.budget)
            cognitive_component = np.random.rand(self.population_size, self.dim)
            social_component = np.random.rand(self.population_size, self.dim)
            self.velocities = (
                inertia_weight * self.velocities
                + cognitive_component * adaptive_rate * (self.personal_best - self.particles)
                + social_component * adaptive_rate * (self.global_best - self.particles)
            )
            self.particles += self.velocities
            self.particles = np.clip(self.particles, lower_bound, upper_bound)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                radius = (1 - (eval_count / self.budget)) * 0.1
                trial_vector = self.variable_neighborhood_search(self.particles[i], radius, lower_bound, upper_bound)
                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best[i] = trial_vector.copy()
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best = trial_vector.copy()

            if (eval_count / self.budget) > 0.5:
                self.population_size = max(10, int(self.initial_population_size * (1 - (eval_count / self.budget))))

        return self.global_best