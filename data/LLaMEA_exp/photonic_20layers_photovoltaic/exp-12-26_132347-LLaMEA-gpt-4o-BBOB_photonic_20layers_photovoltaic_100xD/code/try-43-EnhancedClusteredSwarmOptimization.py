import numpy as np
from sklearn.cluster import KMeans

class EnhancedClusteredSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.best_agent = None
        self.evaluations = 0
        self.history = []
        self.elite_fraction = 0.2

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.evaluations += self.population_size
        self.history.append(self.best_agent)

    def differential_evolution(self, target_idx, func, fitness):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), lb, ub)
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True

        trial = np.where(cross_points, mutant, self.population[target_idx])
        trial_fitness = func(trial)
        if trial_fitness < fitness[target_idx]:
            self.population[target_idx] = trial
            fitness[target_idx] = trial_fitness
            if trial_fitness < func(self.best_agent):
                self.best_agent = trial

    def local_search(self, agent, func):
        noise_scale = 0.1 * (1 - (self.evaluations / self.budget))
        noise = np.random.normal(0, noise_scale, self.dim)
        new_agent = np.clip(agent + noise, func.bounds.lb, func.bounds.ub)
        return new_agent

    def adaptive_clustering(self, func, fitness):
        n_clusters = max(2, self.population_size // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.population)
        for cluster_idx in range(n_clusters):
            cluster_points = self.population[kmeans.labels_ == cluster_idx]
            cluster_center = np.mean(cluster_points, axis=0)
            if len(cluster_points) > 0:
                random_point = cluster_points[np.random.randint(len(cluster_points))]
                new_point = 0.5 * (cluster_center + random_point)
                new_point = np.clip(new_point, func.bounds.lb, func.bounds.ub)
                new_fitness = func(new_point)
                self.evaluations += 1
                if new_fitness < np.min(fitness[kmeans.labels_ == cluster_idx]):
                    min_idx = np.argmin(fitness[kmeans.labels_ == cluster_idx])
                    self.population[kmeans.labels_ == cluster_idx][min_idx] = new_point
                    fitness[kmeans.labels_ == cluster_idx][min_idx] = new_fitness
                    if new_fitness < func(self.best_agent):
                        self.best_agent = new_point

    def elite_preservation_and_recombination(self, func, fitness):
        elite_count = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(fitness)[:elite_count]
        elites = self.population[elite_indices]
        
        new_agents = []
        for i in range(elite_count):
            for j in range(i + 1, elite_count):
                recombined = 0.5 * (elites[i] + elites[j])
                recombined = np.clip(recombined, func.bounds.lb, func.bounds.ub)
                new_agents.append(recombined)
        
        new_agents = np.array(new_agents)
        for agent in new_agents:
            agent_fitness = func(agent)
            self.evaluations += 1
            if agent_fitness < np.max(fitness):
                replace_idx = np.argmax(fitness)
                self.population[replace_idx] = agent
                fitness[replace_idx] = agent_fitness
                if agent_fitness < func(self.best_agent):
                    self.best_agent = agent

    def __call__(self, func):
        self.initialize_population(func)
        fitness = np.array([func(ind) for ind in self.population])

        while self.evaluations < self.budget:
            self.adaptive_clustering(func, fitness)
            self.elite_preservation_and_recombination(func, fitness)
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                self.differential_evolution(idx, func, fitness)
                self.evaluations += 1

                if self.evaluations < self.budget:
                    candidate = self.local_search(self.population[idx], func)
                    candidate_fitness = func(candidate)
                    if candidate_fitness < fitness[idx]:
                        self.population[idx] = candidate
                        fitness[idx] = candidate_fitness
                        self.evaluations += 1
                        if candidate_fitness < func(self.best_agent):
                            self.best_agent = candidate

        return self.best_agent