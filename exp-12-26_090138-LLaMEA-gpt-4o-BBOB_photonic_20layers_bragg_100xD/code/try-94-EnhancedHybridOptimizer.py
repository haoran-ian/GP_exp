import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        population_size = 20 * self.dim
        F_base = 0.6
        CR_base = 0.8
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        best_mem = []
        success_mem = []
        archive = []

        while eval_count < self.budget:
            if eval_count % (population_size) == 0:
                diversity = np.mean(np.std(population, axis=0))
                if diversity < 0.05:
                    population_size = max(self.dim, int(0.9 * population_size))
                else:
                    population_size = min(30 * self.dim, int(1.2 * population_size))

            for i in range(population_size):
                if success_mem:
                    F = np.mean(success_mem) + np.random.normal(0, 0.1)
                    F = np.clip(F, 0.4, 1.0)
                else:
                    F = F_base + np.random.uniform(-0.1, 0.3)

                CR = CR_base + np.random.uniform(-0.1, 0.1)

                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                if len(archive) > 0 and np.random.rand() < 0.3:
                    archive_indx = np.random.randint(len(archive))
                    mutant = np.clip(a + F * (b - c + archive[archive_indx] - a), bounds[:, 0], bounds[:, 1])
                else:
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
                    if len(success_mem) > 10:
                        success_mem.pop(0)
                    if len(best_mem) < 10 or f_trial < np.max(best_mem):
                        best_mem.append(f_trial)
                        best_mem = sorted(best_mem)[:10]
                        archive.append(trial)

                if eval_count < self.budget and np.random.rand() < 0.4:
                    perturbation_scale = 0.1 * np.random.uniform(0.5, 1.5)
                    perturbation = np.random.normal(0, perturbation_scale, self.dim)
                    new_trial = population[i] + perturbation
                    new_trial = np.clip(new_trial, bounds[:, 0], bounds[:, 1])
                    f_new_trial = func(new_trial)
                    eval_count += 1
                    if f_new_trial < f_trial:
                        population[i] = new_trial
                        fitness[i] = f_new_trial
                        if len(best_mem) < 10 or f_new_trial < np.max(best_mem):
                            best_mem.append(f_new_trial)
                            best_mem = sorted(best_mem)[:10]
                            archive.append(new_trial)

        best_idx = np.argmin(fitness)
        return population[best_idx]