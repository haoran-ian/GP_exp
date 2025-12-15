import numpy as np

class EnhancedQuantumBatAlgorithmWithDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.frequency_min = 0
        self.frequency_max = 3
        self.loudness = np.random.uniform(0.5, 1.0, self.population_size)
        self.pulse_rate = np.random.uniform(0.2, 0.8, self.population_size)
        self.alpha = 0.95
        self.gamma = 0.9
        self.beta = np.random.uniform(0, 1, self.population_size)
        self.personal_best = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.exploration_exploitation_tradeoff = 0.5
        self.chaos_factor = 0.1
        self.inertia_weight = 0.9
        self.inertia_decay = 0.99
        self.F = 0.8  # Scaling factor for DE mutation
        self.CR = 0.9  # Crossover rate for DE

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                 (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / lam)
        return step

    def chaotic_perturbation(self, solution, lb, ub):
        chaotic_factor = np.random.uniform(-self.chaos_factor, self.chaos_factor, self.dim)
        perturbed_solution = solution + chaotic_factor * (ub - lb)
        return np.clip(perturbed_solution, lb, ub)

    def quantum_random_walk(self, best_solution, lb, ub):
        q = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0, 1, self.dim)
        return np.clip(best_solution + step_size * q * (ub - lb), lb, ub)

    def differential_mutation(self, population, i, lb, ub):
        idxs = [idx for idx in range(self.population_size) if idx != i]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = np.clip(population[a] + self.F * (population[b] - population[c]), lb, ub)
        return mutant

    def differential_crossover(self, target, mutant):
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.CR:
                trial[j] = mutant[j]
        return trial

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                self.beta[i] = np.random.uniform(0, 1)
                frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * self.beta[i]
                velocities[i] *= self.inertia_weight
                velocities[i] += (population[i] - best_solution) * frequency
                candidate = population[i] + velocities[i]
                candidate = np.clip(candidate, lb, ub)

                if np.random.rand() > self.pulse_rate[i]:
                    candidate = best_solution + self.levy_flight() * self.loudness[i]
                candidate_fitness = func(candidate)
                eval_count += 1

                if candidate_fitness < fitness[i] and np.random.rand() < self.loudness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness

            for i in range(self.population_size):
                mutant = self.differential_mutation(population, i, lb, ub)
                trial = self.differential_crossover(population[i], mutant)
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            self.inertia_weight *= self.inertia_decay
            
        return best_solution