import numpy as np
from sklearn.cluster import KMeans

class EnhancedHybridDEASA_v7:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = 10 * self.dim
        F = 0.8
        CR = 0.9
        T0 = 1000
        Tf = 1e-2
        mem_size = 5
        mem_F = np.full(mem_size, F)
        mem_CR = np.full(mem_size, CR)
        s_F = []
        s_CR = []
        weights = []

        # Initialize a population randomly within the bounds
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += population_size

        # Memory and elite archive
        memory = []
        elite_indices = np.argsort(fitness)[:max(2, population_size // 10)]
        elites = population[elite_indices]

        while self.evaluations < self.budget:
            # Dynamic adjustment of population size
            population_size = max(4, int(10 * self.dim * (1 - self.evaluations / self.budget)))

            # Self-adaptive mechanism for parameter adjustment
            idx = self.evaluations % mem_size
            F = mem_F[idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            CR = mem_CR[idx] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            F = np.clip(F, 0.1, 0.9)
            CR = np.clip(CR, 0.0, 1.0)

            # Apply clustering to adaptively select mutation strategies
            num_clusters = int(np.sqrt(population_size))
            kmeans = KMeans(n_clusters=num_clusters).fit(population)
            labels = kmeans.labels_

            for cluster in range(num_clusters):
                cluster_indices = np.where(labels == cluster)[0]
                cluster_size = len(cluster_indices)

                if cluster_size < 3:
                    continue

                for i in cluster_indices:
                    indices = np.random.choice(cluster_indices, 3, replace=False)
                    x1, x2, x3 = population[indices]
                    scaling_factor = 0.5 + 0.1 * np.sin(self.evaluations / self.budget * np.pi)

                    if np.random.rand() < 0.5:
                        mutant = np.clip(x1 + scaling_factor * (x2 - x3), lb, ub)
                    else:
                        # Introduce elite-guided mutation strategy
                        elite = elites[np.random.randint(len(elites))]
                        mutant = np.clip(x1 + scaling_factor * (elite - x3), lb, ub)

                    trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                    trial_fitness = func(trial)
                    self.evaluations += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        memory.append(trial)
                        s_F.append(F)
                        s_CR.append(CR)
                        weights.append(trial_fitness - fitness[i])

                    if self.evaluations >= self.budget:
                        break

            T = T0 * (Tf / T0) ** (self.evaluations / self.budget)
            for i in range(population_size):
                variance = max(1e-5, np.mean(np.std(population, axis=0)))
                neighbor = population[i] + np.random.normal(0, variance, self.dim) * variance
                neighbor = np.clip(neighbor, lb, ub)
                neighbor_fitness = func(neighbor)
                self.evaluations += 1

                if neighbor_fitness < fitness[i] or np.random.rand() < np.exp(-(neighbor_fitness - fitness[i]) / T):
                    population[i] = neighbor
                    fitness[i] = neighbor_fitness

                if self.evaluations >= self.budget:
                    break
            
            # Update memory with successful F and CR
            if s_F:
                mem_F[idx] = np.sum(np.array(weights) * np.array(s_F)) / np.sum(weights)
                mem_CR[idx] = np.sum(np.array(weights) * np.array(s_CR)) / np.sum(weights)

            # Update elite solutions
            elite_indices = np.argsort(fitness)[:max(2, population_size // 10)]
            elites = population[elite_indices]

            s_F.clear()
            s_CR.clear()
            weights.clear()

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]