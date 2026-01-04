import numpy as np
from sklearn.cluster import KMeans

class AdvancedEnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9
        self.f_min = 0.4
        self.f_max = 0.9
        self.init_temperature = 100
        self.temperature = self.init_temperature
        self.eval_count = 0
        self.cooling_rate = 0.99
        self.min_population_size = 5 * dim
        self.restart_threshold = 0.1 * self.budget
        self.restart_count = 0

    def opposition_based_learning(self, pop, lb, ub):
        opposite_pop = lb + ub - pop
        return np.clip(opposite_pop, lb, ub)

    def adaptive_cooling_schedule(self):
        self.temperature = self.init_temperature * (1 - self.eval_count / self.budget)

    def adaptive_f(self):
        return self.f_min + (self.f_max - self.f_min) * np.exp(-5 * self.eval_count / self.budget)

    def stochastic_ranking(self, population, fitness):
        sort_idx = np.argsort(fitness)
        return population[sort_idx], fitness[sort_idx]

    def resize_population(self, population, fitness, lb, ub):
        new_size = max(self.min_population_size, int(self.population_size * (1 - self.eval_count / self.budget)))
        if new_size < population.shape[0]:
            indices = np.argsort(fitness)[:new_size]
            return population[indices], fitness[indices]
        else:
            extra_size = new_size - population.shape[0]
            extra_pop = np.random.rand(extra_size, self.dim) * (ub - lb) + lb
            extra_fitness = np.array([func(ind) for ind in extra_pop])
            self.eval_count += extra_size
            return np.vstack((population, extra_pop)), np.hstack((fitness, extra_fitness))

    def adaptive_local_search(self, individual, func, lb, ub):
        perturbation = (ub - lb) * 0.05
        trial = np.clip(individual + np.random.uniform(-perturbation, perturbation, self.dim), lb, ub)
        trial_fitness = func(trial)
        self.eval_count += 1
        return trial, trial_fitness

    def intelligent_restart(self, lb, ub):
        self.restart_count += 1
        return np.random.rand(self.population_size, self.dim) * (ub - lb) + lb

    def cluster_population(self, population, n_clusters=3):
        if population.shape[0] < n_clusters:
            return population, np.arange(population.shape[0])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(population)
        return population, kmeans.labels_

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            new_population = []
            for i in range(population.shape[0]):
                idxs = [idx for idx in range(population.shape[0]) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_f = self.adaptive_f()
                mutant = np.clip(a + adaptive_f * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.cr
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / self.temperature):
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if self.eval_count >= self.budget:
                    break

            self.adaptive_cooling_schedule()
            opposite_population = self.opposition_based_learning(new_population, lb, ub)
            opposite_fitness = np.array([func(ind) for ind in opposite_population])
            self.eval_count += len(new_population)

            for j in range(len(new_population)):
                if opposite_fitness[j] < fitness[j]:
                    new_population[j] = opposite_population[j]
                    fitness[j] = opposite_fitness[j]

            population, fitness = self.stochastic_ranking(np.array(new_population), fitness)
            population, fitness = self.resize_population(population, fitness, lb, ub)

            # Apply local search adaptively
            for i in range(population.shape[0]):
                new_population[i], new_fitness = self.adaptive_local_search(population[i], func, lb, ub)
                if new_fitness < fitness[i]:
                    new_population[i] = new_population[i]
                    fitness[i] = new_fitness

            # Cluster population and restart if necessary
            _, labels = self.cluster_population(population)
            if len(set(labels)) < 3 and self.eval_count - self.restart_count * self.restart_threshold > self.restart_threshold:
                population = self.intelligent_restart(lb, ub)
                fitness = np.array([func(ind) for ind in population])
                self.eval_count += self.population_size

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]