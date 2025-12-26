import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.success_history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_individual = pop[best_idx]
        best_fitness = fitness[best_idx]
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    self.success_history.append((self.F, self.CR))
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial

            # Update F and CR based on success history
            if len(self.success_history) > 0:
                mean_F = np.mean([s[0] for s in self.success_history])
                mean_CR = np.mean([s[1] for s in self.success_history])
                self.F = np.clip(mean_F + 0.1 * (np.random.rand() - 0.5), 0.1, 1.0)
                self.CR = np.clip(mean_CR + 0.1 * (np.random.rand() - 0.5), 0.1, 1.0)

            # Dynamic Population Resizing
            if eval_count < self.budget and len(self.success_history) > self.pop_size / 2:
                self.pop_size = max(4, int(self.pop_size * 0.9))  # Reduce population size by 10%
                pop = pop[:self.pop_size]
                fitness = fitness[:self.pop_size]

        return best_individual, best_fitness