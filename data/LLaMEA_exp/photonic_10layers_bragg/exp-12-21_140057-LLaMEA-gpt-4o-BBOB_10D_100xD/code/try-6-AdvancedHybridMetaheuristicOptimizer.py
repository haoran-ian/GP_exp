import numpy as np
import collections

class AdvancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialization
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = population_size
        
        # Parameters
        F, Cr = 0.8, 0.9  # DE parameters
        temp, cooling_rate = 1.0, 0.99  # SA parameters
        neighborhood_radius = 0.1
        memory_size = 5
        memory = collections.deque(maxlen=memory_size)  # Adaptive Memory for best solutions
        
        def differential_evolution(pop, fit):
            indices = np.arange(population_size)
            for i in indices:
                a, b, c = np.random.choice(indices[indices != i], 3, replace=False)
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)
                cross_points = np.random.rand(self.dim) < Cr + 0.05
                trial = np.where(cross_points, mutant, pop[i])
                trial_fitness = func(trial)

                if trial_fitness < fit[i]:
                    pop[i], fit[i] = trial, trial_fitness

        def adaptive_neighborhood_search(ind, fit):
            new_solution = ind + np.random.normal(0, neighborhood_radius, self.dim)
            new_solution = np.clip(new_solution, lb, ub)
            new_fitness = func(new_solution)
            if new_fitness < fit:
                memory.append(new_solution)
                return new_solution, new_fitness
            return ind, fit

        # Search loop
        while evaluations < self.budget:
            # Adaptive adjustment
            F = max(0.5, F * (0.99 + 0.02 * np.random.rand()))
            Cr = min(1.0, Cr * (0.99 + 0.02 * np.random.rand()))

            differential_evolution(population, fitness)
            evaluations += population_size

            # Adaptive Neighborhood Search and Diversity Preservation
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                population[i], fitness[i] = adaptive_neighborhood_search(population[i], fitness[i])
                if fitness[i] < best_fitness:
                    best_solution, best_fitness = population[i], fitness[i]

                # Simulated Annealing Step
                new_solution = population[i] + np.random.normal(0, 0.1, self.dim)
                new_solution = np.clip(new_solution, lb, ub)
                new_fitness = func(new_solution)
                delta = new_fitness - fitness[i]
                
                if delta < 0 or np.exp(-delta / temp) > np.random.rand():
                    population[i], fitness[i] = new_solution, new_fitness
                    if new_fitness < best_fitness:
                        best_solution, best_fitness = new_solution, new_fitness

                evaluations += 1

            # Utilize adaptive memory to refine search
            if memory:
                for sol in memory:
                    perturbed_solution = sol + np.random.normal(0, neighborhood_radius, self.dim)
                    perturbed_solution = np.clip(perturbed_solution, lb, ub)
                    perturbed_fitness = func(perturbed_solution)
                    if perturbed_fitness < best_fitness:
                        best_solution, best_fitness = perturbed_solution, perturbed_fitness

            # Update cooling schedule and neighborhood radius
            temp *= cooling_rate
            neighborhood_radius *= 0.99

        return best_solution