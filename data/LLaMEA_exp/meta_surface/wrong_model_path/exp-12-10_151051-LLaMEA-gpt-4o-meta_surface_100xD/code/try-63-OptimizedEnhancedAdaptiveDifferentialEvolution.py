import numpy as np

class OptimizedEnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        initial_pop_size = 10 * self.dim
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(initial_pop_size, self.dim)
        for i in range(self.dim):
            population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
        
        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        eval_count = initial_pop_size
        
        # Adaptive population resizing parameters
        max_pop_size = 15 * self.dim
        min_pop_size = 5 * self.dim
        resize_threshold = 0.1 * self.budget

        # Optimization loop
        while eval_count < self.budget:
            # Adaptation of parameters
            F = np.random.uniform(0.4, 0.9)
            CR = np.random.uniform(0.3, 0.9)  # Dynamic control of crossover rate

            # Resize population based on progress
            if eval_count % resize_threshold == 0 and eval_count < self.budget:
                # Simple heuristic: halve the population size if progress stalls
                current_pop_size = len(population)
                if current_pop_size > min_pop_size:
                    population = population[:current_pop_size // 2]
                    fitness = fitness[:current_pop_size // 2]

            for i in range(len(population)):
                if eval_count >= self.budget:
                    break

                # Multi-phase mutation strategy
                best_index = np.argmin(fitness)
                second_best_index = fitness.argsort()[1]
                x_best = population[best_index]
                x_second_best = population[second_best_index]
                
                indices = np.random.choice(len(population), 3, replace=False)
                x0, x1, x2 = population[indices]
                if np.random.rand() < 0.5:
                    mutant = np.clip(x0 + F * (x_best - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                else:
                    mutant = np.clip(x0 + F * (x_second_best - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1

                # Greedy local search improvement
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    local_perturb = np.random.normal(0, 0.01, self.dim)
                    perturbed_trial = np.clip(trial + local_perturb, bounds[:, 0], bounds[:, 1])
                    f_perturbed_trial = func(perturbed_trial)
                    eval_count += 1
                    if f_perturbed_trial < f_trial:
                        population[i] = perturbed_trial
                        fitness[i] = f_perturbed_trial

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]