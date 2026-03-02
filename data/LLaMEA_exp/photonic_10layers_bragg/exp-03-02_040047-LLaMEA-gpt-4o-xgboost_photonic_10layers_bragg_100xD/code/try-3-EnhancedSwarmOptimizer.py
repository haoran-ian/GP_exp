import numpy as np

class EnhancedSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Starting population size
        self.inertia_weight = 0.7  # Inertia weight for velocity update
        self.cognitive_coeff = 1.5  # Cognitive coefficient
        self.social_coeff = 1.5  # Social coefficient

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        # Initialize personal bests
        pbest_positions = np.copy(population)
        pbest_fitness = np.copy(fitness)
        
        # Initialize global best
        gbest_index = np.argmin(fitness)
        gbest_position = population[gbest_index]
        gbest_fitness = fitness[gbest_index]

        while evaluations < self.budget:
            # Update velocity and position
            r1, r2 = np.random.rand(2)
            velocity = (self.inertia_weight * velocity +
                        self.cognitive_coeff * r1 * (pbest_positions - population) +
                        self.social_coeff * r2 * (gbest_position - population))
            population = np.clip(population + velocity, lb, ub)
            
            # Evaluate new fitness
            fitness = np.apply_along_axis(func, 1, population)
            evaluations += self.population_size

            # Update personal bests
            better_pbest_mask = fitness < pbest_fitness
            pbest_positions[better_pbest_mask] = population[better_pbest_mask]
            pbest_fitness[better_pbest_mask] = fitness[better_pbest_mask]

            # Update global best
            if np.min(fitness) < gbest_fitness:
                gbest_index = np.argmin(fitness)
                gbest_fitness = fitness[gbest_index]
                gbest_position = population[gbest_index]

            # Dynamic topology adjustment
            if evaluations % 100 == 0:  # Adjust every 100 evaluations
                self.inertia_weight *= 0.99  # Decrease inertia over time
            
        return gbest_position