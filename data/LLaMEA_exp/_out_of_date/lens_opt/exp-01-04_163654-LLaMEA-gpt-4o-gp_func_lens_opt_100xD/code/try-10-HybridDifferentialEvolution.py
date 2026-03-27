import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.CR = 0.9
        self.learning_rate = 0.5
        self.arm_history = []
        self.success_rates = np.zeros(self.pop_size)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.pop_size
        epsilon = 0.1

        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                arm_success_rate = self.success_rates[i] / (np.sum(self.arm_history) + 1)
                exploration_factor = epsilon / (1 + np.sum(self.arm_history))
                if np.random.rand() < arm_success_rate + exploration_factor:
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                else:
                    a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]

                mutant = np.clip(a + self.learning_rate * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.success_rates[i] += 1
                self.arm_history.append(1 if trial_fitness < fitness[i] else 0)

                if num_evaluations >= self.budget:
                    break

            population = new_population
            self.adjust_learning_rate()
            self.update_bandit_strategy()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def adjust_learning_rate(self):
        self.learning_rate = max(0.1, self.learning_rate * 0.999)

    def update_bandit_strategy(self):
        total_arms = np.sum(self.arm_history)
        if total_arms > 0:
            self.success_rates = self.success_rates / total_arms