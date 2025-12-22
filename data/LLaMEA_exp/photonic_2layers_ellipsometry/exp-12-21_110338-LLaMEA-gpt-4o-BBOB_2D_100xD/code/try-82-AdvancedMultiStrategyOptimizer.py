import numpy as np

class AdvancedMultiStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.strategies = [
            self.differential_evolution, 
            self.particle_swarm, 
            self.simulated_annealing,
            self.chaotic_levy_search
        ]
        self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
        self.performance_history = np.zeros(len(self.strategies))
        self.chaos_factor_multiplier = 5  # Adaptive chaos
        self.energy_levels = np.zeros(len(self.strategies))  # Added energy levels

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            chosen_strategy_idx = self.select_strategy()
            chosen_strategy = self.strategies[chosen_strategy_idx]
            new_population, new_fitness = chosen_strategy(population, fitness, func, bounds)
            evals += len(new_fitness)
            
            if evals > self.budget:
                excess = evals - self.budget
                new_population = new_population[:-excess]
                new_fitness = new_fitness[:-excess]
                evals = self.budget

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population, fitness = combined_population[best_indices], combined_fitness[best_indices]

            self.update_strategy_weights(new_fitness, chosen_strategy_idx)

        return population[np.argmin(fitness)]

    def select_strategy(self):
        c = np.random.rand()
        chaotic_factor = (self.chaos_factor_multiplier * c * (1 - c))
        energy_adjusted_weights = self.strategy_weights * (1 + self.energy_levels)  # Adjust weights
        chaos_strategy = np.argmax(chaotic_factor * energy_adjusted_weights)  # Use adjusted weights
        return chaos_strategy

    def update_strategy_weights(self, new_fitness, strategy_idx):
        improvement = np.maximum(0, np.min(new_fitness) - np.min(self.performance_history)) / (np.min(new_fitness) + 1e-6)
        self.performance_history[strategy_idx] += improvement
        total_improvement = np.sum(self.performance_history) + 1e-6
        self.strategy_weights = (self.performance_history / total_improvement)
        self.strategy_weights = self.strategy_weights / self.strategy_weights.sum()
        # Update energy levels based on improvement
        self.energy_levels[strategy_idx] = 1 if improvement > 0 else self.energy_levels[strategy_idx] * 0.9

    def differential_evolution(self, population, fitness, func, bounds):
        F = 0.8 + 0.3 * np.random.rand() * (1 - np.mean(fitness) / np.max(fitness))
        F_adaptive = F * (1 + np.var(population, axis=0).mean())  # Altered line
        CR = 0.7 + 0.2 * np.random.rand()
        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant = np.clip(x1 + F_adaptive * (x2 - x3), bounds[:, 0], bounds[:, 1])
            crossover = np.random.rand(self.dim) < CR
            trial = np.where(crossover, mutant, population[i])
            new_population[i] = trial
            new_fitness[i] = func(trial)
        return new_population, new_fitness

    def particle_swarm(self, population, fitness, func, bounds):
        c1, c2 = 2.05, 2.05
        # Change: use adaptive inertia weight w
        w_min, w_max = 0.4, 0.9
        v_max = 0.2 * (bounds[:, 1] - bounds[:, 0])
        velocities = np.random.uniform(-v_max, v_max, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best = population[np.argmin(fitness)]

        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            w = w_max - ((w_max - w_min) * (i / self.population_size))  # Line changed
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best[i] - population[i]) +
                             c2 * r2 * (global_best - population[i]))
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            new_population[i] = np.clip(population[i] + velocities[i], bounds[:, 0], bounds[:, 1])
            fit = func(new_population[i])
            new_fitness[i] = fit
            if fit < personal_best_fitness[i]:
                personal_best[i], personal_best_fitness[i] = new_population[i], fit
            if fit < func(global_best):
                global_best = new_population[i]
        return new_population, new_fitness

    def simulated_annealing(self, population, fitness, func, bounds):
        T0 = 1000
        alpha = 0.995
        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        for i, x in enumerate(population):
            T = T0
            current_solution = x
            current_fitness = fitness[i]
            for _ in range(10):
                mutation_strength = 0.1 * (T / T0)  # Adaptive mutation strength
                new_solution = current_solution + np.random.normal(0, mutation_strength, self.dim)  # Line changed
                new_solution = np.clip(new_solution, bounds[:, 0], bounds[:, 1])
                new_fitness_candidate = func(new_solution)
                if new_fitness_candidate < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness_candidate) / T):
                    current_solution, current_fitness = new_solution, new_fitness_candidate
                T *= alpha
            new_population[i] = current_solution
            new_fitness[i] = current_fitness
        return new_population, new_fitness

    def chaotic_levy_search(self, population, fitness, func, bounds):
        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        beta = 1.5
        for i in range(self.population_size):
            u = np.random.normal(0, 1, self.dim)
            v = np.random.normal(0, 1, self.dim)
            levy_step = u / (np.abs(v) ** (1 / beta))
            scale = 0.02 * (1 - np.min(fitness) / (np.max(fitness) + 1e-6))
            perturbation = levy_step * scale * (bounds[:, 1] - bounds[:, 0])
            trial = np.clip(population[i] + perturbation, bounds[:, 0], bounds[:, 1])
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                new_population[i], new_fitness[i] = trial, trial_fitness
            else:
                new_population[i], new_fitness[i] = population[i], fitness[i]
        return new_population, new_fitness