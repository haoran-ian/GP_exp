import numpy as np

class EnhancedMemoryAdaptiveDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.F_min = 0.3
        self.F_max = 0.9
        self.CR = 0.9
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.elite_fraction = 0.1
        self.eval_count = 0
        self.convergence_rate = []
        self.memory = np.full(self.pop_size, np.inf)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        pop_fitness = np.array([func(ind) for ind in pop])
        self.eval_count += self.pop_size

        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = np.copy(pop)
        personal_best_fitness = np.copy(pop_fitness)

        global_best_idx = np.argmin(pop_fitness)
        global_best = np.copy(pop[global_best_idx])
        global_best_fitness = pop_fitness[global_best_idx]

        while self.eval_count < self.budget:
            F = self.F_min + (self.F_max - self.F_min) * (1 - (self.eval_count / self.budget))
            elite_size = int(self.pop_size * self.elite_fraction)
            elite_indices = np.argsort(pop_fitness)[:elite_size]
            elites = pop[elite_indices]

            # Adaptive exploration and exploitation strategy
            if len(self.convergence_rate) > 1:
                rate_change = self.convergence_rate[-1] - self.convergence_rate[-2]
                if rate_change > 0:
                    self.w_max = max(self.w_min, self.w_max * 0.98)
                else:
                    self.w_max = min(0.95, self.w_max * 1.02)

            self.convergence_rate.append(global_best_fitness)

            for i in range(self.pop_size):
                if i not in elite_indices:
                    idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                    a, b, c = pop[idxs]
                    memory_factor = np.random.rand() < 0.5
                    mutant = np.clip(a + F * (b - c) + 0.5 * (elites[np.random.randint(elite_size)] - a), lb, ub)
                    if memory_factor and self.memory[i] != np.inf:
                        mutant = np.clip(mutant + 0.5 * (self.memory[i] - mutant), lb, ub)
                    cross_points = np.random.rand(self.dim) < self.CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, pop[i])

                    trial_fitness = func(trial)
                    self.eval_count += 1
                    if trial_fitness < pop_fitness[i]:
                        if trial_fitness < self.memory[i]:
                            self.memory[i] = trial_fitness
                        pop[i] = trial
                        pop_fitness[i] = trial_fitness
                        if trial_fitness < personal_best_fitness[i]:
                            personal_best[i] = trial
                            personal_best_fitness[i] = trial_fitness
                            if trial_fitness < global_best_fitness:
                                global_best = trial
                                global_best_fitness = trial_fitness

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                self.w = self.w_max * (0.99 ** self.eval_count)
                random_perturbation = np.random.normal(0, 0.1 * (1 - self.eval_count / self.budget) * np.exp(-global_best_fitness), self.dim)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best[i] - pop[i]) +
                               self.c2 * r2 * (global_best - pop[i])) + random_perturbation
                pop[i] = np.clip(pop[i] + velocity[i], lb, ub)

                current_fitness = func(pop[i])
                self.eval_count += 1
                if current_fitness < pop_fitness[i]:
                    pop_fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[i]:
                        personal_best[i] = pop[i]
                        personal_best_fitness[i] = current_fitness
                        if current_fitness < global_best_fitness:
                            global_best = pop[i]
                            global_best_fitness = current_fitness

        return global_best, global_best_fitness