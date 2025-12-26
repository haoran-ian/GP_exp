import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize population
        population_size = 20
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        # Differential Evolution parameters
        F = 0.5  # Differential weight
        CR = 0.9  # Crossover probability

        # Particle Swarm Optimization parameters
        velocities = np.random.uniform(-1, 1, (population_size, self.dim))
        pbest_positions = population.copy()
        pbest_fitness = fitness.copy()
        gbest_position = population[np.argmin(fitness)].copy()
        gbest_fitness = np.min(fitness)

        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Particle Swarm Optimization Update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = w * velocities[i] + c1 * r1 * (pbest_positions[i] - population[i]) + c2 * r2 * (gbest_position - population[i])
                population[i] += velocities[i]
                population[i] = np.clip(population[i], func.bounds.lb, func.bounds.ub)

                # Evaluate new fitness
                current_fitness = func(population[i])
                evaluations += 1

                # Update personal best
                if current_fitness < pbest_fitness[i]:
                    pbest_positions[i] = population[i]
                    pbest_fitness[i] = current_fitness

                # Update global best
                if current_fitness < gbest_fitness:
                    gbest_position = population[i]
                    gbest_fitness = current_fitness

                # Differential Evolution Mutation and Crossover
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Adaptive parameter adjustment
            if evaluations < self.budget:
                F = 0.5 + 0.5 * (gbest_fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-10)

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]