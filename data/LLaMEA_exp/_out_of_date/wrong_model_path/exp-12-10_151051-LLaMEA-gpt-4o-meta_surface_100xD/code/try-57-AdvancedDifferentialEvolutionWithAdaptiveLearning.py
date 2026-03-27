import numpy as np

class AdvancedDifferentialEvolutionWithAdaptiveLearning:
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
        
        # Initialize fitness and strategy adaptation parameters
        fitness = np.array([func(ind) for ind in population])
        eval_count = pop_size
        F_memory = np.full(pop_size, 0.5)  # Memory of mutation factors
        CR_memory = np.full(pop_size, 0.5) # Memory of crossover rates

        # Optimization loop
        while eval_count < self.budget:
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Dynamic parameter adaptation
                F = np.clip(F_memory[i] + 0.1 * np.random.standard_normal(), 0.4, 0.9)
                CR = np.clip(CR_memory[i] + 0.1 * np.random.standard_normal(), 0.3, 0.9)
                
                # Mutation strategy
                best_index = np.argmin(fitness)
                x_best = population[best_index]
                
                indices = np.random.choice([idx for idx in range(pop_size) if idx != i], 3, replace=False)
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
                    F_memory[i] = F  # Update memory with successful mutation factor
                    CR_memory[i] = CR # Update memory with successful crossover rate

                    # Adaptive local search improvement
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