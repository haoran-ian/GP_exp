import numpy as np

class HybridChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(100, budget // 10)
        self.mutation_factor_start, self.mutation_factor_end = 0.9, 0.5
        self.crossover_prob_start, self.crossover_prob_end = 0.9, 0.2
        self.global_best_position = None
        self.global_best_value = float('inf')
    
    def chaotically_perturb(self, vector, lb, ub, progress):
        beta = 0.7 # increased perturbation intensity
        chaos_factor = 4 * progress * (1 - progress) # logistic map
        perturbation = beta * chaos_factor * (np.random.rand(self.dim) - 0.5)
        return np.clip(vector + perturbation, lb, ub)
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_positions = population.copy()
        personal_best_values = np.full(self.population_size, float('inf'))

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                current_value = func(population[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = population[i]

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = population[i]

            progress = evaluations / self.budget
            mutation_factor = self.mutation_factor_start * (1 - progress) + self.mutation_factor_end * progress
            crossover_prob = self.crossover_prob_start * (1 - progress) + self.crossover_prob_end * progress

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                trial_vector = population[a] + mutation_factor * (population[b] - population[c])
                trial_vector = self.chaotically_perturb(trial_vector, lb, ub, progress)
                
                crossover_mask = np.random.rand(self.dim) < crossover_prob
                offspring = np.where(crossover_mask, trial_vector, population[i])
                offspring_value = func(offspring)
                evaluations += 1

                if offspring_value < personal_best_values[i]:
                    personal_best_values[i] = offspring_value
                    personal_best_positions[i] = offspring

                if offspring_value < self.global_best_value:
                    self.global_best_value = offspring_value
                    self.global_best_position = offspring

        return self.global_best_position, self.global_best_value