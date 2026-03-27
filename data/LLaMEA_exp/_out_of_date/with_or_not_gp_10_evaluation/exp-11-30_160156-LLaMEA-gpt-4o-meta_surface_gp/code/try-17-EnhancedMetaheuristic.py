import numpy as np

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
    def __call__(self, func):
        population_size = 10
        mutation_factor = 0.8
        crossover_rate = 0.9
        temperature = 1000.0
        cooling_rate = 0.99
        inertia_weight = 0.5  # New line: Inertia weight for PSO
        cognitive_coeff = 1.5  # New line: Cognitive coefficient for PSO
        social_coeff = 2.0  # New line: Social coefficient for PSO
        
        def clip(x):
            return np.clip(x, self.lower_bound, self.upper_bound)
        
        def differential_evolution_step(population):
            new_population = np.copy(population)
            velocities = np.random.uniform(-1, 1, (population_size, self.dim))  # New line: Initial velocities for PSO
            personal_best_positions = np.copy(population)  # New line: Personal best positions for PSO
            personal_best_fitness = np.full(population_size, np.inf)  # New line: Personal best fitness for PSO
            global_best_position = np.copy(population[0])  # New line: Global best position for PSO
            global_best_fitness = np.inf  # New line: Global best fitness for PSO
            
            for i in range(population_size):
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                mutant = clip(population[a] + mutation_factor * (population[b] - population[c]))
                cross_points = np.random.rand(self.dim) < crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                new_population[i] = trial
                
                # PSO update
                r1, r2 = np.random.rand(), np.random.rand()  # New line: Random coefficients for PSO
                velocities[i] = (inertia_weight * velocities[i] + cognitive_coeff * r1 * (personal_best_positions[i] - population[i])
                                 + social_coeff * r2 * (global_best_position - population[i]))  # New line: PSO velocity update
                new_population[i] = clip(population[i] + velocities[i])  # New line: PSO position update
                
            return new_population
        
        def simulated_annealing_step(individual, current_value):
            neighbor = clip(individual + np.random.normal(0, 0.1, self.dim))
            neighbor_value = func(neighbor)
            if neighbor_value < current_value or np.random.rand() < np.exp((current_value - neighbor_value) / temperature):
                return neighbor, neighbor_value
            return individual, current_value
        
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        for i in range(population_size):  # New line: Initialize personal bests
            personal_best_positions[i] = population[i]
            personal_best_fitness[i] = fitness[i]
            if fitness[i] < global_best_fitness:
                global_best_fitness = fitness[i]
                global_best_position = population[i]
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while evaluations < self.budget:
            population = differential_evolution_step(population)
            new_fitness = np.array([func(ind) for ind in population])
            evaluations += population_size
            
            for i in range(population_size):  # New line: Update personal and global bests
                if new_fitness[i] < personal_best_fitness[i]:
                    personal_best_fitness[i] = new_fitness[i]
                    personal_best_positions[i] = population[i]
                if new_fitness[i] < global_best_fitness:
                    global_best_fitness = new_fitness[i]
                    global_best_position = population[i]
            
            current_best_idx = np.argmin(new_fitness)
            if new_fitness[current_best_idx] < best_fitness:
                best_fitness = new_fitness[current_best_idx]
                best = population[current_best_idx]
            
            for i in range(population_size):
                population[i], new_fitness[i] = simulated_annealing_step(population[i], new_fitness[i])
                evaluations += 1
                if new_fitness[i] < best_fitness:
                    best_fitness = new_fitness[i]
                    best = population[i]
                    
            temperature *= cooling_rate
            
        return best