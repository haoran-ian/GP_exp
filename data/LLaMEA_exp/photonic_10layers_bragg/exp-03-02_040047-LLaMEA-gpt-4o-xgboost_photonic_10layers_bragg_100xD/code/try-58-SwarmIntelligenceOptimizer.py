import numpy as np

class SwarmIntelligenceOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_fraction = 0.2
        self.lévy_exponent = 1.5
        self.chaos_control_factor = 0.5
        self.particle_inertia = 0.9
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0
        self.scaling_factor = 0.8
        self.contextual_awareness_factor = 0.9
        self.memory_size = 5

    def chaotic_sequence(self, size):
        x = np.random.rand()
        chaotic_seq = np.zeros(size)
        for i in range(size):
            x = 4 * x * (1 - x)
            chaotic_seq[i] = x
        return chaotic_seq

    def lévy_flight(self, size):
        u = np.random.normal(0, 1, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / self.lévy_exponent)
        return step

    def dynamic_adjustment(self, evaluations, budget):
        progress = evaluations / budget
        return self.contextual_awareness_factor * (1 - progress)

    def adaptive_memory(self, fitness_history):
        if len(fitness_history) < self.memory_size:
            return np.mean(fitness_history)
        else:
            return np.mean(fitness_history[-self.memory_size:])

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size
        personal_best_positions = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_position = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)
        fitness_history = [global_best_fitness]

        while evaluations < self.budget:
            chaotic_seq = self.chaotic_sequence(self.population_size)
            lévy_steps = self.lévy_flight(self.population_size)

            for i in range(self.population_size):
                if chaotic_seq[i] < self.chaos_control_factor:
                    velocities[i] = (self.particle_inertia * velocities[i] +
                                    self.cognitive_coefficient * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                                    self.social_coefficient * np.random.rand(self.dim) * (global_best_position - population[i]))
                    population[i] = np.clip(population[i] + velocities[i], lb, ub)
                else:
                    population[i] = np.clip(global_best_position + lévy_steps[i] * np.random.normal(0, 0.1, self.dim), lb, ub)

            new_fitness = np.apply_along_axis(func, 1, population)
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if new_fitness[i] < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = new_fitness[i]
            
            if np.min(new_fitness) < global_best_fitness:
                global_best_fitness = np.min(new_fitness)
                global_best_position = population[np.argmin(new_fitness)]
                self.scaling_factor += self.dynamic_adjustment(evaluations, self.budget)

            fitness_history.append(global_best_fitness)
            if evaluations % (self.population_size * 5) == 0:
                self.scaling_factor = self.adaptive_memory(fitness_history)

            self.scaling_factor = np.std(fitness) * 0.5

        return global_best_position