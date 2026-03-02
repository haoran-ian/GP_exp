import numpy as np

class EnhancedMultiPopulationMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + 3 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.neighborhood_size = max(5, int(self.population_size * 0.1))
        self.elite_fraction = 0.1
        self.num_subpopulations = 4  # Number of subpopulations for better exploration
        self.subpopulations = [np.random.uniform(0, 1, (self.population_size, self.dim)) for _ in range(self.num_subpopulations)]
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        evaluations = 0
        global_best_fitness = float('inf')
        global_best_solution = None

        while evaluations < self.budget:
            for pop_index in range(self.num_subpopulations):
                population = self.subpopulations[pop_index]
                population = bounds[0] + (bounds[1] - bounds[0]) * population
                fitness = np.array([func(ind) for ind in population])
                evaluations += self.population_size

                elite_size = max(1, int(self.population_size * self.elite_fraction))
                elite_indices = np.argsort(fitness)[:elite_size]

                new_population = np.empty_like(population)
                new_fitness = np.empty(self.population_size)
                
                # Preserve elites
                new_population[:elite_size] = population[elite_indices]
                new_fitness[:elite_size] = fitness[elite_indices]

                for i in range(elite_size, self.population_size):
                    if evaluations >= self.budget:
                        break
                    
                    # Selection of random distinct indices
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    
                    # Mutation and Crossover with adaptive mutation factor
                    self.mutation_factor = 0.5 + 0.5 * np.random.rand()
                    mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                    mutant = np.clip(mutant, bounds[0], bounds[1])
                    cross_points = np.random.rand(self.dim) < self.crossover_probability
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    
                    # Evaluate the trial solution
                    trial_fitness = func(trial)
                    evaluations += 1
                    
                    # Elitist selection
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness

                # Update population with new solutions
                self.subpopulations[pop_index] = (new_population - bounds[0]) / (bounds[1] - bounds[0])

                # Check and update global best
                best_idx = np.argmin(new_fitness)
                if new_fitness[best_idx] < global_best_fitness:
                    global_best_fitness = new_fitness[best_idx]
                    global_best_solution = new_population[best_idx]

                # Dynamic Neighborhood Search with Gaussian perturbation
                for i in range(self.population_size):
                    neighborhood_indices = np.random.choice(
                        self.population_size, self.neighborhood_size, replace=False)
                    neighborhood = new_population[neighborhood_indices]
                    neighborhood_fitness = new_fitness[neighborhood_indices]
                    
                    local_best_idx = np.argmin(neighborhood_fitness)
                    if evaluations >= self.budget:
                        break
                    
                    # Gaussian perturbation around the local best
                    perturbation = np.random.normal(0, 0.1, self.dim)
                    candidate = neighborhood[local_best_idx] + perturbation
                    candidate = np.clip(candidate, bounds[0], bounds[1])
                    candidate_fitness = func(candidate)
                    evaluations += 1

                    # Accept the candidate if it improves the current solution
                    if candidate_fitness < new_fitness[i]:
                        new_population[i] = candidate
                        new_fitness[i] = candidate_fitness

            # Update subpopulation with the best solutions
            for pop_index in range(self.num_subpopulations):
                self.subpopulations[pop_index] = (self.subpopulations[pop_index] - bounds[0]) / (bounds[1] - bounds[0])

        return global_best_solution, global_best_fitness