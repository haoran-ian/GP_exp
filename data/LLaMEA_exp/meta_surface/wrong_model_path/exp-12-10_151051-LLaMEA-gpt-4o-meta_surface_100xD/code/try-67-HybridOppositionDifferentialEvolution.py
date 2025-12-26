import numpy as np

class HybridOppositionDifferentialEvolution:
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
        
        # Initial fitness evaluation
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size

        # Optimization loop
        while eval_count < self.budget:
            # Parameter adaptation
            F = np.random.uniform(0.5, 0.9)  # Slightly higher mutation factor for exploration
            CR = np.random.uniform(0.3, 0.9)

            # Elite opposition-based learning
            best_index = np.argmin(fitness)
            elite = population[best_index]

            opposite_population = bounds[:, 0] + bounds[:, 1] - population
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            eval_count += pop_size
            combined_population = np.vstack((population, opposite_population))
            combined_fitness = np.hstack((fitness, opposite_fitness))

            sorted_indices = np.argsort(combined_fitness)
            population = combined_population[sorted_indices[:pop_size]]
            fitness = combined_fitness[sorted_indices[:pop_size]]

            # Differential Evolution with adaptive local search
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F * (elite - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                eval_count += 1

                # Adaptive local search
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