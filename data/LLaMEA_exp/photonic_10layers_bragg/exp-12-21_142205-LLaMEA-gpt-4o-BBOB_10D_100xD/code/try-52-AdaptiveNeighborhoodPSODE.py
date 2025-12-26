import numpy as np

class AdaptiveNeighborhoodPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, self.budget // 5)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.neighborhood_size = max(2, self.population_size // 10)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        eval_count = self.population_size

        while eval_count < self.budget:
            # Adaptive neighborhood selection
            neighbors = np.random.choice(self.population_size, (self.population_size, self.neighborhood_size), replace=True)

            for i in range(self.population_size):
                # Neighborhood best
                neighborhood_scores = personal_best_scores[neighbors[i]]
                neighborhood_best_idx = neighbors[i][np.argmin(neighborhood_scores)]
                neighborhood_best_position = personal_best_positions[neighborhood_best_idx]

                # PSO Component with neighborhood best
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - population[i]) +
                                 self.social_coeff * r2 * (neighborhood_best_position - population[i]))
                population[i] = np.clip(population[i] + velocities[i], lb, ub)

            # DE Component
            for i in range(self.population_size):
                candidates = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[candidates]
                mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])
                trial_score = func(trial)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score

                    if trial_score < global_best_score:
                        global_best_position = trial
                        global_best_score = trial_score

                if eval_count >= self.budget:
                    break
        
        return global_best_position, global_best_score