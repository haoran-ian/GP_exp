import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
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
        
        # Initialize archive for diversity
        archive = []
        eval_count = pop_size
        
        # Optimization loop
        while eval_count < self.budget:
            # Adaptation of parameters with dynamic strategy
            F = 0.5 + np.random.rand() * 0.5  # Dynamic weighting factor
            CR = 0.5 + np.random.rand() * 0.5  # Dynamic crossover probability
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                
                # Mutation strategy using best and archived solutions
                best_index = np.argmin(fitness)
                x_best = population[best_index]
                if archive:
                    archive_choice = archive[np.random.randint(len(archive))]
                    mutation_factor = np.random.rand()
                    mutant = np.clip(population[i] + mutation_factor * (x_best - archive_choice), bounds[:, 0], bounds[:, 1])
                else:
                    indices = np.random.choice(pop_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = np.clip(x0 + F * (x_best - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                else:
                    # Archive inferior solutions for potential future use
                    if len(archive) < pop_size:
                        archive.append(population[i])
                    else:
                        # Randomly replace an archived solution if the archive is full
                        archive[np.random.randint(pop_size)] = population[i]

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]