import numpy as np

class EnhancedSelfAdaptiveSwarmOptimization:
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
        self.stagnation_threshold = 0.1 * self.budget  
        self.last_improvement_evaluation = 0
        self.entropy_threshold = 0.1 
        self.covariance_matrix = np.identity(dim)  
        self.mean_vector = np.zeros(dim)  
        self.elite_memory = []
        self.elite_fraction = 0.1  

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.last_improvement_evaluation = self.evaluations
        self.evaluations += self.population_size

    def entropy_based_adaptation(self, fitness):
        prob = fitness / np.sum(fitness)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        if entropy < self.entropy_threshold:
            self.F *= 1.03
            self.CR *= 0.95
        else:
            self.F *= 0.97
            self.CR *= 1.05

    def covariance_matrix_adaptation(self, fitness):
        improvement = np.mean(fitness) - np.min(fitness)
        if improvement > 0:
            self.covariance_matrix *= 1.05
        else:
            self.covariance_matrix *= 0.95

    def dynamic_population_resizing(self, fitness):
        if self.evaluations - self.last_improvement_evaluation > self.stagnation_threshold:
            self.population_size = max(10, int(self.population_size / 1.2))
            self.stagnation_threshold *= 1.5
        self.entropy_based_adaptation(fitness)
        self.update_elite_memory(fitness)

    def update_elite_memory(self, fitness):
        elite_size = int(self.elite_fraction * self.population_size)
        sorted_indices = np.argsort(fitness)
        self.elite_memory = self.population[sorted_indices[:elite_size]]

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
                self.last_improvement_evaluation = self.evaluations

    def local_search(self, agent, func):
        noise_scale = max(0.01, 0.1 * (1 - (self.evaluations / self.budget)))  
        noise = np.random.normal(0, noise_scale, self.dim)
        new_agent = np.clip(agent + noise, func.bounds.lb, func.bounds.ub)
        return new_agent

    def covariance_guided_search(self, agent, func):
        z = np.random.multivariate_normal(self.mean_vector, self.covariance_matrix)
        new_agent = np.clip(agent + z, func.bounds.lb, func.bounds.ub)
        return new_agent

    def elite_memetic_search(self, agent, func):
        if self.elite_memory:
            elite_agent = self.elite_memory[np.random.randint(len(self.elite_memory))]
            mutation_vector = np.random.normal(0, 0.1, self.dim)
            new_agent = np.clip(agent + 0.5 * (elite_agent - agent) + mutation_vector, func.bounds.lb, func.bounds.ub)
            return new_agent
        return agent

    def __call__(self, func):
        self.initialize_population(func)
        fitness = np.array([func(ind) for ind in self.population])

        while self.evaluations < self.budget:
            self.dynamic_population_resizing(fitness)
            self.covariance_matrix_adaptation(fitness)
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
                            self.last_improvement_evaluation = self.evaluations

                if self.evaluations < self.budget:
                    candidate = self.covariance_guided_search(self.population[idx], func)
                    candidate_fitness = func(candidate)
                    if candidate_fitness < fitness[idx]:
                        self.population[idx] = candidate
                        fitness[idx] = candidate_fitness
                        self.evaluations += 1
                        if candidate_fitness < func(self.best_agent):
                            self.best_agent = candidate
                            self.last_improvement_evaluation = self.evaluations

                if self.evaluations < self.budget:
                    candidate = self.elite_memetic_search(self.population[idx], func)
                    candidate_fitness = func(candidate)
                    if candidate_fitness < fitness[idx]:
                        self.population[idx] = candidate
                        fitness[idx] = candidate_fitness
                        self.evaluations += 1
                        if candidate_fitness < func(self.best_agent):
                            self.best_agent = candidate
                            self.last_improvement_evaluation = self.evaluations

        return self.best_agent