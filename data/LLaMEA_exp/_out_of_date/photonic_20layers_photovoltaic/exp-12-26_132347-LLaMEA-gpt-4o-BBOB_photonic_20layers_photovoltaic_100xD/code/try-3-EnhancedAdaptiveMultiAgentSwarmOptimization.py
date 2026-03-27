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
        self.memory = []  # Memory to store past best solutions

    def initialize_population(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = [func(ind) for ind in self.population]
        self.best_agent = self.population[np.argmin(fitness)]
        self.memory.append(self.best_agent)  # Store initial best in memory
        self.last_improvement_evaluation = self.evaluations

    def adaptive_parameters(self):
        # Stochastic adaptation of parameters
        self.F = np.random.normal(0.5, 0.2)
        self.CR = np.random.normal(0.9, 0.05)

    def dynamic_population_resizing(self):
        # Decrease population size if improvement stagnates
        if self.evaluations - self.last_improvement_evaluation > self.stagnation_threshold:
            self.population_size = max(10, self.population_size - 5)
            self.stagnation_threshold *= 1.5  # Increase threshold to avoid frequent resizing

    def differential_evolution(self, target_idx, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), lb, ub)

        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True

        trial = np.where(cross_points, mutant, self.population[target_idx])
        return trial

    def local_search(self, agent, func):
        # Enhanced local search with memory influence
        memory_influence = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dim)
        noise = np.random.normal(0, 0.1, self.dim)
        new_agent = np.clip(agent + noise + memory_influence, func.bounds.lb, func.bounds.ub)
        return new_agent

    def __call__(self, func):
        self.initialize_population(func)

        while self.evaluations < self.budget:
            self.adaptive_parameters()
            self.dynamic_population_resizing()
            for idx in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Differential Evolution step
                trial = self.differential_evolution(idx, func)
                if func(trial) < func(self.population[idx]):
                    self.population[idx] = trial
                    self.evaluations += 1
                    if func(trial) < func(self.best_agent):
                        self.best_agent = trial
                        self.last_improvement_evaluation = self.evaluations
                        self.memory.append(self.best_agent)  # Update memory with new best

                # Local search step
                if self.evaluations < self.budget:
                    candidate = self.local_search(self.population[idx], func)
                    if func(candidate) < func(self.population[idx]):
                        self.population[idx] = candidate
                        self.evaluations += 1
                        if func(candidate) < func(self.best_agent):
                            self.best_agent = candidate
                            self.last_improvement_evaluation = self.evaluations
                            self.memory.append(self.best_agent)  # Update memory with new best

        return self.best_agent