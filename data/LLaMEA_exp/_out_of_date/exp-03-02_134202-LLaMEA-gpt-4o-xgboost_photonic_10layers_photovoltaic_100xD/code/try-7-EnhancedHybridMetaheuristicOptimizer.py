import numpy as np

class EnhancedHybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + 3 * dim
        self.mutation_factor = 0.5
        self.crossover_probability = 0.9
        self.neighborhood_size = max(5, int(self.population_size * 0.1))
        self.elite_fraction = 0.1  # Percentage of best individuals to keep as elite
    
    def levy_flight(self, size):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)))**(1/beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        step = u / np.power(np.abs(v), 1/beta)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        elite_size = max(1, int(self.population_size * self.elite_fraction))  # Ensure at least one elite
        elite_indices = np.argsort(fitness)[:elite_size]
        
        while evaluations < self.budget:
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
                
                # Mutation and Crossover
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(cross_points):  # Ensure at least one crossover point
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate the trial solution
                trial_fitness = func(trial)
                evaluations += 1
                
                # Elitist selection
                new_population[i] = trial
                new_fitness[i] = trial_fitness
            
            # Update population with new solutions
            population = new_population
            fitness = new_fitness
            
            # Dynamic Neighborhood Search with Lévy flight perturbation
            for i in range(self.population_size):
                neighborhood_indices = np.random.choice(
                    self.population_size, self.neighborhood_size, replace=False)
                neighborhood = population[neighborhood_indices]
                neighborhood_fitness = fitness[neighborhood_indices]
                
                local_best_idx = np.argmin(neighborhood_fitness)
                if evaluations >= self.budget:
                    break
                
                # Lévy perturbation around the local best
                perturbation = self.levy_flight(self.dim)
                candidate = neighborhood[local_best_idx] + perturbation
                candidate = np.clip(candidate, bounds[0], bounds[1])
                candidate_fitness = func(candidate)
                evaluations += 1

                # Accept the candidate if it improves the current solution
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

            # Update elites
            elite_indices = np.argsort(fitness)[:elite_size]

        # Return the best-found solution
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]