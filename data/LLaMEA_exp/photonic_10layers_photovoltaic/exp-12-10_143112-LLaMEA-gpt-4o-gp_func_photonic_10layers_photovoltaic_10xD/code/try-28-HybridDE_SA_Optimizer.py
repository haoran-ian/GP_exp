import numpy as np

class HybridDE_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.f = 0.8  # Differential mutation factor
        self.cr = 0.9  # Crossover probability
        self.initial_temp = 1.0  # Initial temperature for simulated annealing
        self.cooling_rate = 0.95  # Cooling rate for simulated annealing

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        temperature = self.initial_temp

        while evaluations < self.budget:
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

                if trial_fit < fitness[i] or np.exp((fitness[i] - trial_fit) / temperature) > np.random.rand():
                    population[i] = trial
                    fitness[i] = trial_fit

                # Simulated annealing for exploitation
                for _ in range(5):
                    if evaluations >= self.budget:
                        break

                    neighbor = trial + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_fit = func(neighbor)
                    evaluations += 1

                    if neighbor_fit < trial_fit or np.exp((trial_fit - neighbor_fit) / temperature) > np.random.rand():
                        trial = neighbor
                        trial_fit = neighbor_fit

                population[i] = trial
                fitness[i] = trial_fit

            temperature *= self.cooling_rate

        best_idx = np.argmin(fitness)
        return population[best_idx]