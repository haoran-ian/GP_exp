import numpy as np

class AdvancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = 30
        F_init = 0.8
        CR_init = 0.9
        T0 = 1.0
        alpha = 0.98
        elitism_rate = 0.2
        niche_radius = 0.1
        num_clusters = 3

        # Initialize population
        population = np.random.rand(pop_size, self.dim)
        for i in range(pop_size):
            population[i] = func.bounds.lb + population[i] * (func.bounds.ub - func.bounds.lb)

        # Evaluate initial population
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        temperature = T0

        def distance(x, y):
            return np.linalg.norm(x - y)

        while num_evaluations < self.budget:
            # Dynamic adjustment of parameters
            F = F_init * (1 - num_evaluations / self.budget)
            CR = CR_init * (1 - num_evaluations / self.budget)

            # Clustering for niche preservation
            clusters = [population[i::num_clusters] for i in range(num_clusters)]
            cluster_fitness = [fitness[i::num_clusters] for i in range(num_clusters)]

            for cluster_idx, (cluster, cluster_fitness) in enumerate(zip(clusters, cluster_fitness)):
                elite_count = int(elitism_rate * len(cluster))
                elite_indices = cluster_fitness.argsort()[:elite_count]
                elites = cluster[elite_indices]

                for i in range(len(cluster)):
                    if i in elite_indices:
                        continue

                    idxs = np.random.choice(len(cluster), 3, replace=False)
                    x0, x1, x2 = cluster[idxs]
                    mutant = x0 + F * (x1 - x2)
                    mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                    crossover_mask = np.random.rand(self.dim) < CR
                    trial = np.where(crossover_mask, mutant, cluster[i])

                    trial_fitness = func(trial)
                    num_evaluations += 1

                    # Simulated Annealing acceptance
                    if trial_fitness < cluster_fitness[i]:
                        cluster[i] = trial
                        cluster_fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_solution = trial
                            best_fitness = trial_fitness
                    else:
                        acceptance_prob = np.exp((cluster_fitness[i] - trial_fitness) / temperature)
                        if np.random.rand() < acceptance_prob:
                            cluster[i] = trial
                            cluster_fitness[i] = trial_fitness

                    if num_evaluations >= self.budget:
                        break

                # Reintroduce elites to maintain diversity within clusters
                cluster[:elite_count] = elites

            # Update global population and fitness
            population = np.vstack(clusters)
            fitness = np.hstack(cluster_fitness)

            temperature *= alpha

        return best_solution