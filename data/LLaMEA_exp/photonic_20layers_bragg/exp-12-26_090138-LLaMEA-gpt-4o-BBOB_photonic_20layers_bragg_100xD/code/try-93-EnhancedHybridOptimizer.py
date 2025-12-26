import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)
        
        # Initialize parameters for Differential Evolution
        population_size = 15 * self.dim
        F_base = 0.6
        CR_base = 0.8
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        
        # Initialize population
        population = np.random.rand(population_size, self.dim)
        population = bounds[:, 0] + population * (bounds[:, 1] - bounds[:, 0])
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Memory for best solutions and past success rate
        best_mem = []
        success_mem = []
        past_success_rate = 0
        
        # Initialize elitist archive
        archive = []

        while eval_count < self.budget:
            # Dynamic resizing based on performance
            if eval_count % (2 * population_size) == 0:  
                diversity = np.mean(np.std(population, axis=0))
                success_rate = len(success_mem) / (population_size * 2)
                if success_rate < past_success_rate:
                    population_size = max(self.dim, int(0.9 * population_size))
                else:
                    population_size = min(20 * self.dim, int(1.1 * population_size))
                past_success_rate = success_rate

            for i in range(population_size):
                # Adaptive parameter selection with learning strategy
                F = F_base + np.random.normal(0, 0.1 * (1 - success_rate))
                CR = CR_base + np.random.normal(0, 0.1 * success_rate)

                # Mutation and crossover
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                if archive and np.random.rand() < 0.2:  # Use archive-guided mutation
                    archive_indx = np.random.randint(len(archive))
                    mutant = np.clip(a + F * (b - c + archive[archive_indx] - a), bounds[:, 0], bounds[:, 1])
                else:
                    mutant = np.clip(a + F * (b - c), bounds[:, 0], bounds[:, 1])
                
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate the trial candidate
                f_trial = func(trial)
                eval_count += 1

                # Selection and success memory update
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    success_mem.append(F)
                    if len(success_mem) > 5:
                        success_mem.pop(0)
                    if len(best_mem) < 5 or f_trial < np.max(best_mem):
                        best_mem.append(f_trial)
                        best_mem = sorted(best_mem)[:5]
                        archive.append(trial)  # Add to archive

                # Ensemble-based local search
                if eval_count < self.budget and np.random.rand() < 0.3:
                    perturbation_scale = 0.05 * np.random.uniform(0.5, 1.5)
                    gaussian_perturbation = np.random.normal(0, perturbation_scale, self.dim)
                    uniform_perturbation = np.random.uniform(-perturbation_scale, perturbation_scale, self.dim)
                    ensemble_perturbation = (gaussian_perturbation + uniform_perturbation) / 2
                    new_trial = population[i] + ensemble_perturbation
                    new_trial = np.clip(new_trial, bounds[:, 0], bounds[:, 1])
                    f_new_trial = func(new_trial)
                    eval_count += 1
                    if f_new_trial < f_trial:
                        population[i] = new_trial
                        fitness[i] = f_new_trial
                        if len(best_mem) < 5 or f_new_trial < np.max(best_mem):
                            best_mem.append(f_new_trial)
                            best_mem = sorted(best_mem)[:5]
                            archive.append(new_trial)  # Add to archive

        # Return the best solution found
        best_idx = np.argmin(fitness)
        return population[best_idx]