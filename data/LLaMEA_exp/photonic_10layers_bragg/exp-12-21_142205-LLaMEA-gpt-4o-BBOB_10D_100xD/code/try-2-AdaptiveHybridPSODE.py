import numpy as np

class AdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, self.budget // 5)
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.elite_fraction = 0.1  # Fraction of elites in the population

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
            # Update inertia weight dynamically
            self.inertia_weight = 0.4 + (0.5 * (self.budget - eval_count) / self.budget)
            
            # Sort population by personal best scores and retain elites
            elite_count = max(1, int(self.elite_fraction * self.population_size))
            elite_indices = np.argsort(personal_best_scores)[:elite_count]
            elites = personal_best_positions[elite_indices]

            # PSO Component
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)

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

            # Replace worst solutions with elites
            worst_indices = np.argsort(personal_best_scores)[-elite_count:]
            population[worst_indices] = elites

        return global_best_position, global_best_score