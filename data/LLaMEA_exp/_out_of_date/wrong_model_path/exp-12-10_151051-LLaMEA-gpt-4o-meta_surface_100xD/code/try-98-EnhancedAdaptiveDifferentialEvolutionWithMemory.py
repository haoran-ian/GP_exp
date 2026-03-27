import numpy as np

class EnhancedAdaptiveDifferentialEvolutionWithMemory:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        # Initialize parameters
        pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(pop_size, self.dim)
        for i in range(self.dim):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size

        memory = np.zeros((pop_size, self.dim))  # Memory to store previous best individuals

        # Optimization loop
        while eval_count < self.budget:
            # Adaptation of parameters
            F = np.random.uniform(0.4, 0.9)
            CR = np.random.uniform(0.3, 0.9)  # Dynamic control of crossover rate

            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Mutation with dynamic strategy based on current best fitness
                best_index = np.argmin(fitness)
                x_best = population[best_index]
                
                # Memory-based selection
                if np.random.rand() < 0.5:
                    x_memory = memory[np.random.randint(pop_size)]
                else:
                    x_memory = x_best
                
                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                
                # Memory-enhanced mutation strategy
                mutant = np.clip(x0 + F * (x_memory - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1
                
                # Adaptive local search intensity
                adaptive_intensity = 0.01 * (1 - eval_count / self.budget)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    # Store the improved individual in memory
                    memory[i] = trial
                    # Apply adaptive local search
                    local_perturb = np.random.normal(0, adaptive_intensity, self.dim)
                    perturbed_trial = np.clip(trial + local_perturb, bounds[:, 0], bounds[:, 1])
                    f_perturbed_trial = func(perturbed_trial)
                    eval_count += 1
                    if f_perturbed_trial < f_trial:
                        population[i] = perturbed_trial
                        fitness[i] = f_perturbed_trial
                        memory[i] = perturbed_trial  # Update memory with the perturbed trial

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]