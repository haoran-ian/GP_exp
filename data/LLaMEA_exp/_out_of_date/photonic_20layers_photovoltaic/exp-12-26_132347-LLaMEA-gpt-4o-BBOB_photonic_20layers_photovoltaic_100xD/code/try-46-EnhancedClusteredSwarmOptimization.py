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
        self.stagnation_threshold = 0.1 * self.budget
        self.last_improvement_evaluation = 0
        self.entropy_threshold = 0.1
        self.covariance_matrix = np.identity(dim)
        self.mean_vector = np.zeros(dim)
        self.elite_fraction = 0.2
        self.elite_enhancement_factor = 1.1

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.last_improvement_evaluation = self.evaluations
        self.evaluations += self.population_size

    def dynamic_population_resizing(self, fitness):
        if self.evaluations - self.last_improvement_evaluation > self.stagnation_threshold:
            self.population_size = max(10, int(self.population_size / 1.2))
            self.stagnation_threshold *= 1.5

    def entropy_based_adaptation(self, fitness):
        prob = fitness / np.sum(fitness)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        if entropy < self.entropy_threshold:
            self.F *= 1.03
            self.CR *= 0.95
        else:
            self.F *= 0.97
            self.CR *= 1.05

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

    def elite_preservation(self, fitness):
        elite_count = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(fitness)[:elite_count]
        return self.population[elite_indices]

    def adaptive_gaussian_perturbation(self, agent, func):
        sigma = np.std(self.population, axis=0) * self.elite_enhancement_factor
        g_noise = np.random.normal(0, sigma, self.dim)
        new_agent = np.clip(agent + g_noise, func.bounds.lb, func.bounds.ub)
        return new_agent

    def elite_enhancement(self, func, elite_population, fitness):
        enhanced_elite = []
        for elite in elite_population:
            new_agent = self.adaptive_gaussian_perturbation(elite, func)
            new_fitness = func(new_agent)
            self.evaluations += 1
            if new_fitness < func(self.best_agent):
                self.best_agent = new_agent
                self.last_improvement_evaluation = self.evaluations
            enhanced_elite.append(new_agent)
        return np.array(enhanced_elite)

    def __call__(self, func):
        self.initialize_population(func)
        fitness = np.array([func(ind) for ind in self.population])
        while self.evaluations < self.budget:
            self.dynamic_population_resizing(fitness)
            self.entropy_based_adaptation(fitness)
            elite_population = self.elite_preservation(fitness)
            enhanced_elite = self.elite_enhancement(func, elite_population, fitness)
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                self.differential_evolution(idx, func, fitness)
                self.evaluations += 1
                if self.evaluations < self.budget:
                    candidate = self.adaptive_gaussian_perturbation(self.population[idx], func)
                    candidate_fitness = func(candidate)
                    if candidate_fitness < fitness[idx]:
                        self.population[idx] = candidate
                        fitness[idx] = candidate_fitness
                        self.evaluations += 1
                        if candidate_fitness < func(self.best_agent):
                            self.best_agent = candidate
                            self.last_improvement_evaluation = self.evaluations

        return self.best_agent