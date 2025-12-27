import numpy as np

class EnhancedAdaptiveMultiAgentSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.F = 0.5  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.population = None
        self.best_agent = None
        self.evaluations = 0
        self.stagnation_threshold = 0.1 * self.budget  # Threshold to trigger dynamic changes
        self.last_improvement_evaluation = 0
        self.entropy_threshold = 0.1  # Threshold for entropy-based adjustments

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in self.population])
        self.best_agent = self.population[np.argmin(fitness)]
        self.last_improvement_evaluation = self.evaluations
        self.evaluations += self.population_size

    def entropy_based_adaptation(self, fitness):
        # Calculate fitness entropy
        prob = fitness / np.sum(fitness)
        entropy = -np.sum(prob * np.log(prob + 1e-10))  # Avoid log(0)
        if entropy < self.entropy_threshold:
            self.F *= 1.03  # Adjusted exploration boost
            self.CR *= 0.95  # Enhance refinement
        else:
            self.F *= 0.95  # Adjusted F reduction factor
            self.CR *= 1.05

    def adaptive_parameters(self):
        # Stochastic adaptation of parameters
        self.F = np.clip(np.random.normal(self.F, 0.1), 0.1, 1.0)
        self.CR = np.clip(np.random.normal(self.CR, 0.05), 0.1, 1.0)

    def dynamic_population_resizing(self, fitness):
        # Adjust population size based on performance
        if self.evaluations - self.last_improvement_evaluation > self.stagnation_threshold:
            self.population_size = max(10, int(self.population_size / 1.2))
            self.stagnation_threshold *= 1.5  # Increase threshold to avoid frequent resizing

        # Entropy-based adaptation
        self.entropy_based_adaptation(fitness)

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
        noise_scale = max(0.01, 0.1 * (1 - (self.evaluations / self.budget)))  # Adaptive noise scaling
        noise = np.random.normal(0, noise_scale, self.dim)
        new_agent = np.clip(agent + noise, func.bounds.lb, func.bounds.ub)
        return new_agent

    def __call__(self, func):
        self.initialize_population(func)
        fitness = np.array([func(ind) for ind in self.population])

        while self.evaluations < self.budget:
            self.adaptive_parameters()
            self.dynamic_population_resizing(fitness)
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Differential Evolution step
                self.differential_evolution(idx, func, fitness)
                self.evaluations += 1

                # Local search step
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

        return self.best_agent