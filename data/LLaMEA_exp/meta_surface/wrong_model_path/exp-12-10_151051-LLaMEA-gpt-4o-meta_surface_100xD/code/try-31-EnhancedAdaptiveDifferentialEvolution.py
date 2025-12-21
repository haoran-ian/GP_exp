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
        eval_count = pop_size
        
        # Archive initialization
        archive = np.empty((0, self.dim))
        
        # Optimization loop
        while eval_count < self.budget:
            # Adaptation of parameters
            F = np.random.uniform(0.4, 0.9)
            CR = np.random.uniform(0.5, 1.0)
            
            # Dynamic population adaptation
            if eval_count % (self.budget // 10) == 0:
                pop_size = max(4, int(pop_size * 0.9))
                population = population[:pop_size]
                fitness = fitness[:pop_size]

            new_population = []
            new_fitness = []
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Mutation with dynamic strategy
                best_index = np.argmin(fitness)
                second_best_index = (fitness.argsort()[1])
                x_best = population[best_index]
                x_second_best = population[second_best_index]
                
                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                
                if np.random.rand() < 0.7:
                    mutant = x0 + F * (x_best - x0) + F * (x1 - x2)
                elif np.random.rand() < 0.85 and len(archive) > 0:
                    rand_idx = np.random.choice(len(archive))
                    x_archive = archive[rand_idx]
                    mutant = x0 + F * (x_archive - x0) + F * (x1 - x2)
                else:
                    mutant = x0 + F * (x_second_best - x0) + F * (x1 - x2)
                
                mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                    archive = np.vstack([archive, population[i]])
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
                
            population = np.array(new_population)
            fitness = np.array(new_fitness)

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]