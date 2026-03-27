import numpy as np

class AdaptiveEnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * self.dim
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Slower cooling for prolonged exploration
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.exploration_factor = 0.1
        self.feedback_factor = 0.05  # New factor for feedback-driven adaptation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.population_size

        best_solution = None
        best_fitness = float('inf')

        while budget_used < self.budget:
            # Differential Evolution with Feedback-Driven Adaptation
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mutation_factor = self.mutation_factor / (1 + 0.01 * budget_used)
                dynamic_exploration = self.exploration_factor * (np.random.rand() - 0.5)
                mutant = np.clip(a + adaptive_mutation_factor * (b - c) + dynamic_exploration, lb, ub)
                dynamic_crossover_rate = self.crossover_rate + 0.05 * np.cos(budget_used / self.budget * np.pi)
                crossover = np.random.rand(self.dim) < dynamic_crossover_rate
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best solution found
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if budget_used >= self.budget:
                    break
            
            # Cool down temperature
            self.temperature *= self.cooling_rate

            # Dynamic adjustment for mutation and exploration factors
            cluster_centers = np.mean(population, axis=0)
            diversity = np.linalg.norm(population - cluster_centers, axis=1).mean()
            if diversity < 0.1 * (ub - lb).mean():
                self.mutation_factor *= 1.2
                self.exploration_factor *= 1.1
            
            # Feedback-driven parameter tuning based on improvement
            if best_fitness < fitness.mean():
                self.mutation_factor *= 1 - self.feedback_factor
                self.exploration_factor *= 1 - self.feedback_factor
            else:
                self.mutation_factor *= 1 + self.feedback_factor
                self.exploration_factor *= 1 + self.feedback_factor

        return best_solution, best_fitness