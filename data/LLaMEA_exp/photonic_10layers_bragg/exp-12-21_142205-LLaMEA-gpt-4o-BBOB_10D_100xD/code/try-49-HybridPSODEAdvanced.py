import numpy as np

class HybridPSODEAdvanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, self.budget // 5)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 2.0
        self.social_coeff = 1.7
        self.mutation_factor = 0.9
        self.crossover_rate = 0.95
        self.chaotic_seq = self._init_chaotic_sequence()
    
    def _init_chaotic_sequence(self):
        # Initialize chaotic sequence using logistic map
        seq = np.zeros(self.budget)
        seq[0] = 0.5  # Initial value
        for i in range(1, self.budget):
            seq[i] = 4 * seq[i-1] * (1 - seq[i-1])
        return seq

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
            # Adaptive Inertia Weight
            self.inertia_weight = 0.4 + 0.3 * (self.budget - eval_count) / self.budget

            # PSO Component with Chaotic Influence
            r1 = self.chaotic_seq[eval_count:self.population_size+eval_count].reshape(-1, 1)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)

            # DE Component with Adaptive Mutation Factor
            adaptive_mutation = self.mutation_factor * (1 - (eval_count / self.budget))
            for i in range(self.population_size):
                candidates = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[candidates]
                mutant = np.clip(x1 + adaptive_mutation * (x2 - x3), lb, ub)
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