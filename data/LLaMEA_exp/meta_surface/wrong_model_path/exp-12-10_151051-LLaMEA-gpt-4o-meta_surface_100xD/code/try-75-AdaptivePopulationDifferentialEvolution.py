import numpy as np

class AdaptivePopulationDifferentialEvolution:
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

        # Optimization loop
        while eval_count < self.budget:
            # Adaptation of parameters
            F = np.random.uniform(0.4, 0.9)
            CR = np.random.uniform(0.3, 0.9)

            # Dynamically adjust population size based on fitness diversity
            if np.std(fitness) < 0.01:  # Small diversity, increase exploration
                pop_size = min(initial_pop_size * 2, self.budget - eval_count)
            else:  # High diversity, focus on exploitation
                pop_size = initial_pop_size

            # Evaluate using a subset of current population if necessary
            current_indices = np.random.choice(len(population), size=pop_size, replace=False)
            current_population = population[current_indices]
            current_fitness = fitness[current_indices]

            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Mutation with dynamic strategy based on current best fitness
                best_index = np.argmin(current_fitness)
                x_best = current_population[best_index]
                
                indices = np.random.choice(pop_size, 3, replace=False)
                x0, x1, x2 = current_population[indices]
                mutant = np.clip(x0 + F * (x_best - x0) + F * (x1 - x2), bounds[:, 0], bounds[:, 1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, current_population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1

                # Greedy local search improvement
                if f_trial < current_fitness[i]:
                    current_population[i] = trial
                    current_fitness[i] = f_trial
                    # Adjust perturbation scale based on budget consumption
                    local_perturb = np.random.normal(0, 0.01 * (1 - eval_count / self.budget), self.dim)
                    perturbed_trial = np.clip(trial + local_perturb, bounds[:, 0], bounds[:, 1])
                    f_perturbed_trial = func(perturbed_trial)
                    eval_count += 1
                    if f_perturbed_trial < f_trial:
                        current_population[i] = perturbed_trial
                        current_fitness[i] = f_perturbed_trial

            # Update the original population with improved candidates
            population[current_indices] = current_population
            fitness[current_indices] = current_fitness

        # Best solution
        best_index = np.argmin(fitness)
        return population[best_index]