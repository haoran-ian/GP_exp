import numpy as np

class HADE_RLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover rate

    def __call__(self, func):
        eval_count = 0

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count += self.population_size

        while eval_count < self.budget:
            # Calculate diversity
            fitness_std = np.std(fitness)
            self.F = 0.5 + fitness_std / (fitness_std + 1) * 0.4
            self.CR = 0.9 * (1 - fitness_std / (fitness_std + 1))
            
            # Differential Evolution
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                
                # Crossover
                trial = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < self.CR
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break

            # Randomized Local Search
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                local_candidate = np.copy(population[i])
                perturbation = np.random.normal(0, 0.1, self.dim)
                local_candidate = np.clip(local_candidate + perturbation, self.lower_bound, self.upper_bound)

                local_fitness = func(local_candidate)
                eval_count += 1
                if local_fitness < fitness[i]:
                    population[i] = local_candidate
                    fitness[i] = local_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]