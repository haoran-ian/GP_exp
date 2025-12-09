import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.max_iterations = budget // self.population_size
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.num_subpopulations = 3  # Number of subpopulations for diverse exploration
        
    def __call__(self, func):
        subpopulations = [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size // self.num_subpopulations, self.dim)) for _ in range(self.num_subpopulations)]
        fitnesses = [np.array([func(ind) for ind in sub]) for sub in subpopulations]
        best_overall_solution = None
        best_overall_fitness = float('inf')

        for iteration in range(self.max_iterations):
            for k in range(self.num_subpopulations):
                population = subpopulations[k]
                fitness = fitnesses[k]
                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]

                for i in range(len(population)):
                    indices = np.random.permutation(len(population))
                    x1, x2, x3 = population[indices[:3]]

                    # Differential Evolution Mutation
                    F = 0.5 + np.random.rand() * 0.3  # Randomized scale factor for diversity
                    mutant = x1 + F * (x2 - x3)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    # Adaptive Crossover
                    adaptive_rate = 0.8 - 0.5 * (iteration / self.max_iterations)  # Adaptation based on iteration
                    crossover_mask = np.random.rand(self.dim) < adaptive_rate
                    trial = np.where(crossover_mask, mutant, population[i])

                    # Simulated Annealing Acceptance
                    trial_fitness = func(trial)
                    if trial_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - trial_fitness) / self.temperature):
                        population[i] = trial
                        fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Update overall best solution
                if best_fitness < best_overall_fitness:
                    best_overall_solution = best_solution
                    best_overall_fitness = best_fitness

                # Periodic migration between subpopulations
                if iteration % 10 == 0:
                    migrants = [np.argmin(fitnesses[k]) for k in range(self.num_subpopulations)]
                    for i in range(self.num_subpopulations):
                        for j in range(self.num_subpopulations):
                            if i != j:
                                subpopulations[j][migrants[j]] = subpopulations[i][migrants[i]]

            # Cooling
            self.temperature *= self.cooling_rate

        return best_overall_solution