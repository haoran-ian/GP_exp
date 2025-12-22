import numpy as np

class EnhancedHybridPSODE_DynamicAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, self.budget // 5)
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.inertia_min = 0.4
        self.mutation_min = 0.5
        self.social_coeff_min = 1.0

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
            # Dynamic PSO Component
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)

            # Adaptive inertia weight and social coefficient
            self.inertia_weight = max(self.inertia_min, 0.9 - 0.5 * (eval_count / self.budget))
            self.social_coeff = max(self.social_coeff_min, 1.5 - 0.5 * (eval_count / self.budget))

            # DE Component with dynamic mutation
            for i in range(self.population_size):
                candidates = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[candidates]
                dynamic_mutation = self.mutation_min + (self.mutation_factor - self.mutation_min) * (self.budget - eval_count) / self.budget
                mutant = np.clip(x1 + dynamic_mutation * (x2 - x3), lb, ub)
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
        
        return global_best_position, global_best_score