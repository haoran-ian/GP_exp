import numpy as np

class EnhancedHybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8  # Differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.local_search_iters = 5
        self.gradient_step_size = 0.01

    def __call__(self, func):
        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Differential evolution - selection, mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), lb, ub)

                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fit = func(trial)
                evaluations += 1

                if trial_fit < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fit

                # Local search with gradient-based refinement
                for _ in range(self.local_search_iters):
                    if evaluations >= self.budget:
                        break

                    # Gradient approximation
                    grad = np.zeros(self.dim)
                    for d in range(self.dim):
                        perturbed = np.copy(trial)
                        perturbed[d] += self.gradient_step_size
                        grad[d] = (func(perturbed) - trial_fit) / self.gradient_step_size
                        evaluations += 1
                        if evaluations >= self.budget:
                            break

                    # Gradient descent step
                    trial = np.clip(trial - self.gradient_step_size * grad, lb, ub)
                    trial_fit = func(trial)
                    evaluations += 1

                    if trial_fit < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fit

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]