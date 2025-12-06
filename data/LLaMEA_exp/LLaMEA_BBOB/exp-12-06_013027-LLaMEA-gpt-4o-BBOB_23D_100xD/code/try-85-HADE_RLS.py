import numpy as np

class HADE_RLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # Mutation factor
        self.CR_init = 0.9  # Initial Crossover rate
        self.CR_final = 0.4  # Final Crossover rate

    def __call__(self, func):
        eval_count = 0

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count += self.population_size

        while eval_count < self.budget:
            # Differential Evolution
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                F_adaptive = self.F * np.random.uniform(0.5, 1.5) * np.exp(-eval_count / self.budget)  
                dimension_scaling = np.random.uniform(0.5, 1.0, self.dim) * (1 - (eval_count / self.budget))  
                mutant = np.clip(x1 + F_adaptive * (x2 - x3) * dimension_scaling, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(population[i])
                CR_adaptive = self.CR_init - (self.CR_init - self.CR_final) * (eval_count / self.budget)
                crossover_points = np.random.rand(self.dim) < CR_adaptive
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
                perturbation_variance = 0.1 + 0.5 * (eval_count / self.budget)  # Changed line
                perturbation = np.random.normal(0, perturbation_variance, self.dim)
                local_candidate = np.clip(local_candidate + perturbation, self.lower_bound, self.upper_bound)

                local_fitness = func(local_candidate)
                eval_count += 1
                if local_fitness < fitness[i]:
                    population[i] = local_candidate
                    fitness[i] = local_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]