import numpy as np

class EnhancedMultiStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.strategies = [
            self.differential_evolution, 
            self.particle_swarm, 
            self.simulated_annealing,
            self.chaotic_search
        ]
        self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
        self.performance_history = np.zeros(len(self.strategies))
        self.elite_fraction = 0.1

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
        chaos = np.random.rand()
        chaotic_factor = (4 * chaos * (1 - chaos))  # Logistic map for stronger chaos
        strategy_probs = self.strategy_weights * chaotic_factor
        strategy_probs /= strategy_probs.sum()
        return np.random.choice(len(self.strategies), p=strategy_probs)

    def update_strategy_weights(self, new_fitness, strategy_idx):
        recent_improvement = np.minimum(0, np.min(new_fitness) - np.min(self.performance_history)) / (np.abs(np.min(new_fitness)) + 1e-6)
        self.performance_history[strategy_idx] += recent_improvement
        self.strategy_weights = self.performance_history / (self.performance_history.sum() + 1e-6)

    def differential_evolution(self, population, fitness, func, bounds):
        F = 0.8
        CR = 0.9
        elite_count = max(1, int(self.elite_fraction * self.population_size))
        elite_indices = np.argsort(fitness)[:elite_count]
        elites = population[elite_indices]

        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            indices = np.random.choice(elite_count, 3, replace=False)
            x1, x2, x3 = elites[indices]
            mutant = np.clip(x1 + F * (x2 - x3), bounds[:, 0], bounds[:, 1])
            crossover = np.random.rand(self.dim) < CR
            trial = np.where(crossover, mutant, population[i])
            new_population[i] = trial
            new_fitness[i] = func(trial)
        return new_population, new_fitness

    def particle_swarm(self, population, fitness, func, bounds):
        c1, c2 = 2.05, 2.05
        w = 0.729
        v_max = 0.2 * (bounds[:, 1] - bounds[:, 0])
        velocities = np.random.uniform(-v_max, v_max, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best = population[np.argmin(fitness)]

        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
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
                new_solution = current_solution + np.random.normal(0, 0.1, self.dim)
                new_solution = np.clip(new_solution, bounds[:, 0], bounds[:, 1])
                new_fitness_candidate = func(new_solution)
                if new_fitness_candidate < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness_candidate) / T):
                    current_solution, current_fitness = new_solution, new_fitness_candidate
                T *= alpha
            new_population[i] = current_solution
            new_fitness[i] = current_fitness
        return new_population, new_fitness

    def chaotic_search(self, population, fitness, func, bounds):
        new_population = np.empty_like(population)
        new_fitness = np.empty(self.population_size)
        for i in range(self.population_size):
            chaotic_sequence = np.random.rand(self.dim)
            scale = 0.02 * (1 - np.min(fitness) / (np.max(fitness) + 1e-6))
            for j in range(5):
                chaotic_sequence = np.sin(np.pi * chaotic_sequence)
                perturbation = (chaotic_sequence - 0.5) * 2 * (bounds[:, 1] - bounds[:, 0]) * scale
                trial = np.clip(population[i] + perturbation, bounds[:, 0], bounds[:, 1])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_population[i], new_fitness[i] = trial, trial_fitness
                else:
                    new_population[i], new_fitness[i] = population[i], fitness[i]
        return new_population, new_fitness