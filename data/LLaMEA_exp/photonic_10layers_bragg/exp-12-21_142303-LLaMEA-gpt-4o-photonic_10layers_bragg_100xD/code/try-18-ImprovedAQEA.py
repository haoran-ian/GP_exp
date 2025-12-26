import numpy as np

class ImprovedAQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(30, dim * 2)  # Dynamic population size based on dimension
        self.q_population = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.rotation_angle = 0.05  # Quantum rotation gate parameter

    def _quantum_observation(self):
        angles = np.arccos(1 - 2 * self.q_population)
        return (np.cos(angles) > np.random.rand(*angles.shape)).astype(float)

    def _evaluate_population(self, func, bounds):
        real_population = bounds.lb + self._quantum_observation() * (bounds.ub - bounds.lb)
        fitness = np.array([func(ind) for ind in real_population])
        return real_population, fitness

    def _update_best(self, real_population, fitness):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = real_population[min_idx]

    def _quantum_rotation_gate(self):
        rotation_matrix = np.array(
            [[np.cos(self.rotation_angle), -np.sin(self.rotation_angle)], 
             [np.sin(self.rotation_angle), np.cos(self.rotation_angle)]]
        )
        for i in range(self.population_size):
            obs = self._quantum_observation()[i]
            angle = np.arccos(1 - 2 * obs)
            new_angle = angle + self.rotation_angle * (2 * np.random.rand(self.dim) - 1)
            self.q_population[i] = (1 - np.cos(new_angle)) / 2

    def _adaptive_mutation(self):
        mutation_strength = np.abs(np.random.normal(0, 0.05, self.q_population.shape))
        adapt_factor = np.random.rand(*self.q_population.shape)
        self.q_population += mutation_strength * (adapt_factor - 0.5)
        self.q_population = np.clip(self.q_population, 0, 1)

    def __call__(self, func):
        bounds = func.bounds
        evaluations = 0

        while evaluations < self.budget:
            real_population, fitness = self._evaluate_population(func, bounds)
            self._update_best(real_population, fitness)
            evaluations += self.population_size
            self._quantum_rotation_gate()  # Apply quantum rotation gate
            self._adaptive_mutation()

        return self.best_solution