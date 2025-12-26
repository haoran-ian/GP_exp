import numpy as np

class EnhancedAdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        population_size = 15 * self.dim
        F_base = 0.6
        CR_base = 0.8
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T

        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        best_mem = []
        success_mem = []
        dynamic_restart_threshold = 50  # Threshold for restart based on memory diversity

        while eval_count < self.budget:
            if eval_count % (2 * population_size) == 0:
                diversity = np.mean(np.std(population, axis=0))
                if diversity < 0.05: 
                    population_size = max(self.dim, int(0.9 * population_size))
                else:
                    population_size = min(20 * self.dim, int(1.1 * population_size))

                # Dynamic restart if memory diversity is low
                if len(best_mem) >= dynamic_restart_threshold and np.std(best_mem) < 0.001:
                    population = np.random.rand(population_size, self.dim)
                    population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
                    fitness = np.array([func(ind) for ind in population])
                    eval_count += population_size
                    best_mem.clear()
                    success_mem.clear()
                    continue

            for i in range(population_size):
                if success_mem:
                    F = np.mean(success_mem)
                    F = np.clip(F + np.random.normal(0, 0.1), 0.4, 1.0)
                else:
                    F = F_base + np.random.uniform(-0.1, 0.3)

                CR = CR_base + np.random.uniform(-0.2, 0.2)

                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_mem.append(F)
                    if len(success_mem) > 5:
                        success_mem.pop(0)
                    if len(best_mem) < dynamic_restart_threshold or f_trial < np.max(best_mem):
                        best_mem.append(f_trial)
                        best_mem = sorted(best_mem)[:dynamic_restart_threshold]

                if eval_count < self.budget and np.random.rand() < 0.3:
                    perturbation_scale = 0.05 * np.random.uniform(0.5, 1.5)
                    perturbation = np.random.normal(0, perturbation_scale, self.dim)
                    new_trial = population[i] + perturbation
                    new_trial = np.clip(new_trial, bounds[:, 0], bounds[:, 1])
                    f_new_trial = func(new_trial)
                    eval_count += 1
                    if f_new_trial < f_trial:
                        population[i] = new_trial
                        fitness[i] = f_new_trial
                        if len(best_mem) < dynamic_restart_threshold or f_new_trial < np.max(best_mem):
                            best_mem.append(f_new_trial)
                            best_mem = sorted(best_mem)[:dynamic_restart_threshold]

        best_idx = np.argmin(fitness)
        return population[best_idx]