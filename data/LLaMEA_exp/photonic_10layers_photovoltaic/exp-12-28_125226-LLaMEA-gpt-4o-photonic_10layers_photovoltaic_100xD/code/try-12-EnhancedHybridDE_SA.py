import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8
        self.initial_temperature = 100
        self.cooling_rate = 0.99
        self.eval_count = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size
        temperature = self.initial_temperature
        stagnation_counter = 0
        best_fitness = np.min(fitness)

        while self.eval_count < self.budget:
            cr = 0.5 + 0.5 * np.random.rand()  # Dynamic crossover rate
            new_population = []

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_f = self.f * (1 - self.eval_count / self.budget) + 0.1
                mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < cr
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / temperature):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if self.eval_count >= self.budget:
                    break

            population = np.array(new_population)
            temperature *= self.cooling_rate

            # Restart mechanism to avoid stagnation
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > self.population_size:
                population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
                fitness = np.array([func(ind) for ind in population])
                best_fitness = np.min(fitness)
                self.eval_count += self.population_size
                stagnation_counter = 0

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]