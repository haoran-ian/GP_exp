import numpy as np

class EnhancedChaoticDEwithAdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.base_crossover_rate = 0.9
        self.population = None
        self.fitness = None
        self.chaos_sequence = self.generate_chaos_sequence(budget)

    def generate_chaos_sequence(self, size):
        chaos_sequence = np.zeros(size)
        chaos_sequence[0] = np.random.rand()
        for i in range(1, size):
            chaos_sequence[i] = 4.0 * chaos_sequence[i-1] * (1.0 - chaos_sequence[i-1])
        return chaos_sequence

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
    
    def evaluate_population(self, func):
        for i, individual in enumerate(self.population):
            if self.fitness[i] == np.inf:  # Evaluate only unevaluated individuals
                self.fitness[i] = func(individual)
    
    def chaotic_differential_evolution(self, func, chaos_index):
        diversity = np.mean(np.std(self.population, axis=0))
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            chaotic_factor = 1 + self.chaos_sequence[chaos_index]
            adaptive_mutation_factor = self.mutation_factor * diversity * (1 - chaos_index/self.budget)
            mutant = np.clip(a + adaptive_mutation_factor * chaotic_factor * (b - c), func.bounds.lb, func.bounds.ub)
            
            adaptive_crossover_rate = self.base_crossover_rate * (1 - diversity)
            cross_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)
            
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness

    def chaotic_particle_swarm_optimization(self, func):
        velocities = np.zeros_like(self.population)
        personal_best_positions = np.copy(self.population)
        personal_best_fitness = np.copy(self.fitness)
        
        global_best_index = np.argmin(self.fitness)
        global_best_position = self.population[global_best_index]
        
        w = 0.5  # inertia weight
        c1 = c2 = 2.0  # cognitive and social coefficients

        for i in range(self.population_size):
            r1, r2 = np.random.rand(2, self.dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - self.population[i]) +
                             c2 * r2 * (global_best_position - self.population[i]))
            new_position = np.clip(self.population[i] + velocities[i], func.bounds.lb, func.bounds.ub)
            new_fitness = func(new_position)
            
            if new_fitness < personal_best_fitness[i]:
                personal_best_positions[i], personal_best_fitness[i] = new_position, new_fitness
                if new_fitness < self.fitness[global_best_index]:
                    global_best_position = new_position

            if new_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = new_position, new_fitness

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        self.evaluate_population(func)
        evaluations = self.population_size
        chaos_index = 0

        while evaluations < self.budget:
            self.chaotic_differential_evolution(func, chaos_index)
            evaluations += self.population_size
            chaos_index = min(chaos_index + self.population_size, len(self.chaos_sequence) - 1)
            
            if evaluations < self.budget:
                self.chaotic_particle_swarm_optimization(func)
                evaluations += self.population_size

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]