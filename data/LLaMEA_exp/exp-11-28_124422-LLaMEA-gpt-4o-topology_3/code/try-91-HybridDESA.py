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

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index].copy()
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Differential Evolution mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                fitness_diversity = np.std(fitness) / (np.mean(fitness) + 1e-9)
                temperature_factor = self.temperature_schedule[eval_count]  # Adding a temperature factor
                self.F = 0.6 + 0.4 * fitness_diversity * temperature_factor + (0.2 * eval_count / self.budget)  # Adjusted scaling factor
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Dynamic crossover probability with elitism
                dynamic_CR = self.CR * (0.5 + 0.5 * fitness_diversity * eval_count / self.budget)  # Updated dynamic crossover
                if np.random.rand() < 0.5:
                    trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, population[i])
                else:
                    trial = np.where(np.random.rand(self.dim) < dynamic_CR, mutant, best_solution)
                trial_fitness = func(trial)
                eval_count += 1

                # Selection with Simulated Annealing acceptance
                if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature_schedule[eval_count - 1]):
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update the best solution
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Adjust population size dynamically
            self.population_size = max(5, int((eval_count / self.budget) * (10 + 5 * self.dim)))

        return best_solution