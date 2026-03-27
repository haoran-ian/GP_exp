import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class EnhancedACDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_range = (0.4, 1.0)
        self.CR_range = (0.1, 0.9)
        self.bounds = None
        self.elite_fraction = 0.1  # Fraction of population considered elite
        self.learning_rate = 0.1  # Adaptive learning rate for parameter tuning

    def initialize_population(self):
        lower, upper = self.bounds.lb, self.bounds.ub
        return np.random.uniform(lower, upper, (self.population_size, self.dim))

    def select_parents(self, population, scores):
        idx = np.random.choice(self.population_size, 3, replace=False)
        return population[idx], scores[idx]

    def mutate(self, target, best, r1, r2, F, strategy='rand1'):
        if strategy == 'rand1':
            return np.clip(target + F * (r1 - r2), self.bounds.lb, self.bounds.ub)
        elif strategy == 'best1':
            return np.clip(best + F * (r1 - r2), self.bounds.lb, self.bounds.ub)
        else:
            raise ValueError("Unknown mutation strategy.")

    def crossover(self, target, mutant, CR):
        crossover_mask = np.random.rand(self.dim) < CR
        return np.where(crossover_mask, mutant, target)

    def adapt_parameters(self, F, CR, success=False):
        if success:
            F = min(self.F_range[1], max(self.F_range[0], F + self.learning_rate * (self.F_range[1] - F)))
            CR = min(self.CR_range[1], max(self.CR_range[0], CR + self.learning_rate * (self.CR_range[1] - CR)))
        else:
            F = max(self.F_range[0], F - self.learning_rate * (F - self.F_range[0]))
            CR = max(self.CR_range[0], CR - self.learning_rate * (CR - self.CR_range[0]))
        return F, CR

    def cluster_population(self, population):
        kmeans = KMeans(n_clusters=min(self.population_size // 20, 5), n_init=1)
        kmeans.fit(population)
        return kmeans.labels_

    def local_search(self, best, func):
        noise = np.random.normal(0, 0.1, self.dim)
        candidate = np.clip(best + noise, self.bounds.lb, self.bounds.ub)
        return candidate, func(candidate)

    def dynamic_population_resize(self, population, scores):
        elite_count = int(np.ceil(self.population_size * self.elite_fraction))
        elite_indices = np.argsort(scores)[:elite_count]
        new_population_size = int(self.population_size * 0.9)
        if new_population_size < elite_count:
            new_population_size = elite_count
        return population[elite_indices][:new_population_size], scores[elite_indices][:new_population_size]

    def __call__(self, func):
        self.bounds = func.bounds
        population = self.initialize_population()
        scores = np.array([func(ind) for ind in population])
        F, CR = np.random.uniform(*self.F_range), np.random.uniform(*self.CR_range)

        evals = self.population_size
        while evals < self.budget:
            population, scores = self.dynamic_population_resize(population, scores)
            clusters = self.cluster_population(population)
            for i in range(len(population)):
                cluster_indices = np.where(clusters == clusters[i])[0]
                if len(cluster_indices) > 1:
                    target = population[i]
                    best_idx = cluster_indices[np.argmin(scores[cluster_indices])]
                    best = population[best_idx]
                    r1, r2 = population[np.random.choice(cluster_indices, 2, replace=False)]
                    mutation_strategy = 'best1' if np.random.rand() < 0.7 else 'rand1'  # Modified line
                    mutant = self.mutate(target, best, r1, r2, F, strategy=mutation_strategy)
                    trial = self.crossover(target, mutant, CR)
                    trial_score = func(trial)
                    evals += 1
                    if trial_score < scores[i]:
                        population[i] = trial
                        scores[i] = trial_score
                        F, CR = self.adapt_parameters(F, CR, success=True)
                    else:
                        F, CR = self.adapt_parameters(F, CR, success=False)
                    if evals >= self.budget:
                        break
            best_idx = np.argmin(scores)
            candidate, candidate_score = self.local_search(population[best_idx], func)
            if candidate_score < scores[best_idx]:
                population[best_idx] = candidate
                scores[best_idx] = candidate_score
                evals += 1

        best_idx = np.argmin(scores)
        return population[best_idx], scores[best_idx]