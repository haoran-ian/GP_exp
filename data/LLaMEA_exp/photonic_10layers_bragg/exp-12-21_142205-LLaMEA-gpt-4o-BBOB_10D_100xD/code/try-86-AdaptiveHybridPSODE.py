import numpy as np

class AdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, self.budget // 5)
        self.inertia_weight = 0.9  # Start with a higher inertia for exploration
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.inertia_decay = 0.99  # Decay factor for inertia
        self.mutation_decay = 0.995  # Decay factor for mutation
        self.cognitive_decay = 0.98  # Decay factor for cognitive coefficient
        self.social_growth = 1.02  # Growth factor for social coefficient

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
            # PSO Component with adaptive inertia and learning rates
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)
            self.inertia_weight *= self.inertia_decay  # Decay the inertia weight
            self.cognitive_coeff *= self.cognitive_decay  # Decay the cognitive coefficient
            self.social_coeff *= self.social_growth  # Grow the social coefficient

            # DE Component with local search
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

                # Local search around the best solution found so far
                if eval_count < self.budget and np.random.rand() < 0.1:
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    local_trial = np.clip(global_best_position + perturbation, lb, ub)
                    local_score = func(local_trial)
                    eval_count += 1
                    if local_score < global_best_score:
                        global_best_position = local_trial
                        global_best_score = local_score

                if eval_count >= self.budget:
                    break

            self.mutation_factor *= self.mutation_decay  # Decay mutation factor
        
        return global_best_position, global_best_score