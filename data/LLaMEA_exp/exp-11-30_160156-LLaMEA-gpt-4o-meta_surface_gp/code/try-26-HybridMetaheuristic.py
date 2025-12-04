import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        initial_population_size = 10
        mutation_factor = 0.8
        crossover_rate = 0.9
        temperature = 1000.0
        cooling_rate = 0.995  # Modified: Adjusted cooling rate
        
        def clip(x):
            return np.clip(x, self.lower_bound, self.upper_bound)
        
        def dynamic_population_size(evaluations):
            max_population = 20  # New line: Maximum population size
            return min(initial_population_size + evaluations // 100, max_population)  # New line: Increase population size over time
        
        def differential_evolution_step(population):
            pop_size = len(population)  # New line: Dynamic population size
            new_population = np.copy(population)
            for i in range(pop_size):
                candidates = list(range(pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                mutant = clip(population[a] + mutation_factor * (population[b] - population[c]))
                cross_points = np.random.rand(self.dim) < crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                new_population[i] = trial
            return new_population
        
        def simulated_annealing_step(individual, current_value):
            neighbor = clip(individual + np.random.normal(0, 0.1, self.dim))
            neighbor_value = func(neighbor)
            if neighbor_value < current_value or np.random.rand() < np.exp((current_value - neighbor_value) / temperature):
                return neighbor, neighbor_value
            return individual, current_value
        
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = initial_population_size
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        
        # Optimization loop
        while evaluations < self.budget:
            # Adjust population size
            population_size = dynamic_population_size(evaluations)  # New line: Adjust population size
            if len(population) < population_size:
                additional_pop = np.random.uniform(self.lower_bound, self.upper_bound, (population_size - len(population), self.dim))
                population = np.vstack((population, additional_pop))
                fitness = np.append(fitness, [func(ind) for ind in additional_pop])
                evaluations += len(additional_pop)
            
            # Differential Evolution step
            population = differential_evolution_step(population)
            
            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in population])
            evaluations += len(population)
            
            # Update best solution
            current_best_idx = np.argmin(new_fitness)
            if new_fitness[current_best_idx] < best_fitness:
                best_fitness = new_fitness[current_best_idx]
                best = population[current_best_idx]
            
            # Simulated Annealing step
            for i in range(len(population)):
                population[i], new_fitness[i] = simulated_annealing_step(population[i], new_fitness[i])
                evaluations += 1
                if new_fitness[i] < best_fitness:
                    best_fitness = new_fitness[i]
                    best = population[i]
                    
            # Apply cooling to temperature
            temperature *= cooling_rate
            
        return best