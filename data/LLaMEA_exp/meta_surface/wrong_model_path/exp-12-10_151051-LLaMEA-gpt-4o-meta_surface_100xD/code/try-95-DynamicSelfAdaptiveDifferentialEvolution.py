import numpy as np

class DynamicSelfAdaptiveDifferentialEvolution:
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

        # Optimization loop
        while eval_count < self.budget:
            # Adaptation of parameters
            F = np.random.uniform(0.4, 0.9)
            CR = np.random.uniform(0.3, 0.9)

            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Dynamic selection of strategy based on population diversity
                best_index = np.argmin(fitness)
                diversity = np.std(population, axis=0).mean()
                adaptive_threshold = 0.1 * (1 - eval_count / self.budget)
                
                if diversity < adaptive_threshold:
                    F = 0.9
                    CR = 0.9
                else:
                    F = 0.4 + 0.5 * (1 - eval_count / self.budget)
                
                # Mutation
                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1

                # Local search with adaptive learning rate
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    # Adjust learning rate based on improvement
                    learning_rate = 0.01 * (1 - eval_count / self.budget)
                    local_perturb = np.random.normal(0, learning_rate, self.dim)
                    perturbed_trial = np.clip(trial + local_perturb, bounds[:, 0], bounds[:, 1])
                    f_perturbed_trial = func(perturbed_trial)
                    eval_count += 1
                    if f_perturbed_trial < f_trial:
                        population[i] = perturbed_trial
                        fitness[i] = f_perturbed_trial

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]