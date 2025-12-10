import numpy as np
from sklearn.cluster import KMeans

class EnhancedAdaptiveDifferentialMigrationWithCooperativeNiching:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 10 + 2 * dim
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, dim))
        self.best = None
        self.learning_rate_initial = 0.05
        self.learning_rate_min = 0.005
        self.eval_count = 0
        self.group_size = max(1, dim // 5)  # Divide dimensions into smaller subgroups for cooperative co-evolution

    def _dynamic_parameters(self, fitness, subgroup):
        fitness_variance = np.var(fitness)
        F = self.learning_rate * np.random.rand() + 0.5 * (1 + 0.1 * fitness_variance) + 0.2 * np.std(fitness) / np.mean(fitness)
        CR = self.learning_rate * np.random.rand() + 0.8
        return F, CR

    def _mutate(self, indices, F, subgroup):
        a, b, c = self.population[indices]
        mutant = a.copy()
        mutant[subgroup] = a[subgroup] + F * (b[subgroup] - c[subgroup]) + 0.1 * (self.best[subgroup] - a[subgroup])
        mutant = np.clip(mutant, self.lb, self.ub)
        return mutant

    def _crossover(self, target, mutant, CR, subgroup):
        crossover = np.random.rand(self.dim) < CR
        crossover[subgroup] = True
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover, mutant, target)
        return trial

    def _selection(self, fitness, trial, trial_fitness, i):
        if trial_fitness < fitness[i]:
            self.population[i] = trial
            fitness[i] = trial_fitness
            if trial_fitness < fitness[np.argmin(fitness)]:
                self.best = trial

    def _evaluate_population(self, func):
        fitness = np.apply_along_axis(func, 1, self.population)
        self.eval_count += self.population_size
        self.best = self.population[np.argmin(fitness)]
        return fitness

    def __call__(self, func):
        fitness = self._evaluate_population(func)

        while self.eval_count < self.budget:
            self.learning_rate = self.learning_rate_initial * (1 - self.eval_count / self.budget) + self.learning_rate_min

            subgroups = [range(i, min(i + self.group_size, self.dim)) for i in range(0, self.dim, self.group_size)]
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break

                for subgroup in subgroups:
                    F, CR = self._dynamic_parameters(fitness, subgroup)
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    mutant = self._mutate(np.random.choice(indices, 3, replace=False), F, subgroup)
                    trial = self._crossover(self.population[i], mutant, CR, subgroup)

                    trial_fitness = func(trial)
                    self.eval_count += 1
                    self._selection(fitness, trial, trial_fitness, i)

            if self.eval_count < self.budget:
                self._dynamic_population_replacement(func, fitness)

            if self.eval_count % (self.budget // 10) == 0:
                reinit_count = self.population_size // 5
                self.population[:reinit_count] = np.random.uniform(self.lb, self.ub, (reinit_count, self.dim))

            if self.eval_count < self.budget:
                self._niching_strategy(func, fitness)

        return self.best

    def _dynamic_population_replacement(self, func, fitness):
        dynamic_sigma = 0.1 + 0.5 * (1 - self.eval_count / self.budget)
        new_population = self.best + np.random.normal(0, dynamic_sigma, (self.population_size, self.dim))
        dynamic_perturbation = 0.7 * (1 - self.eval_count / self.budget)
        perturbed_population = new_population + dynamic_perturbation * np.random.uniform(-1, 1, new_population.shape)
        perturbed_population = np.clip(perturbed_population, self.lb, self.ub)
        new_fitness = np.apply_along_axis(func, 1, perturbed_population)
        self.eval_count += self.population_size

        combined_population = np.vstack((self.population, perturbed_population))
        combined_fitness = np.hstack((fitness, new_fitness))
        best_indices = np.argsort(combined_fitness)[:self.population_size]
        self.population = combined_population[best_indices]
        fitness[:] = combined_fitness[best_indices]
        self.best = self.population[np.argmin(fitness)]

    def _niching_strategy(self, func, fitness):
        num_clusters = max(2, self.population_size // 5)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        labels = kmeans.fit_predict(self.population)

        for cluster_id in range(num_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_pop = self.population[cluster_indices]
            cluster_fitness = fitness[cluster_indices]

            if len(cluster_pop) > 1:
                best_cluster_index = np.argmin(cluster_fitness)
                if cluster_fitness[best_cluster_index] > np.min(fitness):
                    cluster_pop[best_cluster_index] = np.clip(self.best + np.random.normal(0, 0.1, self.dim), self.lb, self.ub)
                    fitness[cluster_indices[best_cluster_index]] = func(cluster_pop[best_cluster_index])
                    self.eval_count += 1