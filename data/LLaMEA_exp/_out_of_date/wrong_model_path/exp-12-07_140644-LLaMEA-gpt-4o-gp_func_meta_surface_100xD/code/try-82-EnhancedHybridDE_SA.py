import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.max_iterations = budget // self.population_size
        self.temperature = 100.0
        self.cooling_rate = 0.98
        self.num_subpopulations = 3

    def __call__(self, func):
        subpopulations = [
            np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size // self.num_subpopulations, self.dim))
            for _ in range(self.num_subpopulations)
        ]
        fitness = [np.array([func(ind) for ind in subpop]) for subpop in subpopulations]
        best_idx = [np.argmin(fit) for fit in fitness]
        best_solutions = [subpop[i] for subpop, i in zip(subpopulations, best_idx)]
        best_fitness = [fit[i] for fit, i in zip(fitness, best_idx)]
        
        global_best_fit_idx = np.argmin(best_fitness)
        global_best_solution = best_solutions[global_best_fit_idx]
        global_best_fitness = best_fitness[global_best_fit_idx]

        for iteration in range(self.max_iterations):
            adaptive_rate = 0.9 - 0.6 * (iteration / self.max_iterations)
            for s in range(self.num_subpopulations):
                for i in range(len(subpopulations[s])):
                    indices = np.random.permutation(len(subpopulations[s]))
                    x1, x2, x3 = subpopulations[s][indices[:3]]

                    # Differential Evolution Mutation
                    mutant = x1 + 0.8 * (x2 - x3)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    # Crossover
                    crossover_mask = np.random.rand(self.dim) < adaptive_rate
                    trial = np.where(crossover_mask, mutant, subpopulations[s][i])

                    # Simulated Annealing Acceptance
                    trial_fitness = func(trial)
                    if trial_fitness < fitness[s][i] or np.random.rand() < np.exp((fitness[s][i] - trial_fitness) / self.temperature):
                        subpopulations[s][i] = trial
                        fitness[s][i] = trial_fitness

                    # Update best solution found in subpopulation
                    if trial_fitness < best_fitness[s]:
                        best_solutions[s] = trial
                        best_fitness[s] = trial_fitness

            # Update global best solution
            for s in range(self.num_subpopulations):
                if best_fitness[s] < global_best_fitness:
                    global_best_solution = best_solutions[s]
                    global_best_fitness = best_fitness[s]

            # Cooling
            self.temperature *= self.cooling_rate

        return global_best_solution