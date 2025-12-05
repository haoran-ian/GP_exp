import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 + 5 * dim
        self.F = 0.9  # Updated Differential weight
        self.CR = 0.9  # Crossover probability
        self.initial_temperature = 1.0
        self.final_temperature = 0.01
        self.temperature_schedule = np.linspace(self.initial_temperature, self.final_temperature, budget)
        self.restart_threshold = budget // 10  # Adaptive restart threshold

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index].copy()
        best_fitness = fitness[best_index]
        last_improvement = 0

        while eval_count < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                fitness_diversity = np.std(fitness) / (np.mean(fitness) + 1e-9)
                temperature_factor = self.temperature_schedule[eval_count] 
                self.F = 0.6 + 0.3 * fitness_diversity * temperature_factor + (0.3 * eval_count / self.budget)
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, self.lb, self.ub)

                dynamic_CR = self.CR * (0.5 + 0.5 * fitness_diversity * eval_count / self.budget)
                if np.random.rand() < 0.5:
                    trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, population[i])
                else:
                    trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, best_solution)
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature_schedule[eval_count - 1]):
                    population[i] = trial
                    fitness[i] = trial_fitness
                    last_improvement = eval_count

                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            if eval_count - last_improvement > self.restart_threshold:
                worst_indices = np.argsort(fitness)[-self.population_size // 3:]
                population[worst_indices] = np.random.uniform(self.lb, self.ub, (len(worst_indices), self.dim))
                fitness[worst_indices] = np.array([func(ind) for ind in population[worst_indices]])
                eval_count += len(worst_indices)

            self.population_size = max(5, int((eval_count / self.budget) * (10 + 5 * self.dim)))

        return best_solution