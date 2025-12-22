import numpy as np

class AdvancedAdaptiveSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.frequency_bounds = (0, 3)
        self.alpha_bounds = (0.5, 0.9)
        self.gamma_bounds = (0.7, 0.95)
        self.loudness = np.random.uniform(0.5, 1.0, self.population_size)
        self.pulse_rate = np.random.uniform(0.2, 0.8, self.population_size)
        self.velocity = np.zeros((self.population_size, self.dim))
        self.memory = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.personal_best = np.copy(self.memory)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.improvement_count = np.zeros(self.population_size)
        self.stagnation_threshold = 5
        self.exploration_exploitation_tradeoff = 0.4
        self.chaos_intensity = 0.1
        self.inertia_weight = 0.9
        self.inertia_decay = 0.95
        self.global_best_history = []

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                 (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / lam)
        return step

    def chaotic_injection(self, solution, lb, ub):
        chaotic_change = np.random.uniform(-self.chaos_intensity, self.chaos_intensity, self.dim)
        return np.clip(solution + chaotic_change * (ub - lb), lb, ub)

    def quantum_tunneling(self, best_solution, lb, ub):
        q = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0, 1, self.dim)
        return np.clip(best_solution + step_size * q * (ub - lb), lb, ub)

    def dynamic_parameter_adjustment(self, improvement_rate):
        phase_adjustment = np.tanh(self.exploration_exploitation_tradeoff * improvement_rate)
        self.alpha = np.clip(self.alpha_bounds[0] + (self.alpha_bounds[1] - self.alpha_bounds[0]) * phase_adjustment, *self.alpha_bounds)
        self.gamma = np.clip(self.gamma_bounds[0] + (self.gamma_bounds[1] - self.gamma_bounds[0]) * phase_adjustment, *self.gamma_bounds)
        self.loudness = np.clip(self.loudness * (1.0 + phase_adjustment), 0.5, 1.0)
        self.pulse_rate = np.clip(self.pulse_rate * (1.0 - phase_adjustment), 0.2, 0.8)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        eval_count = self.population_size
        self.global_best_history.append(best_fitness)

        while eval_count < self.budget:
            for i in range(self.population_size):
                frequency = self.frequency_bounds[0] + (self.frequency_bounds[1] - self.frequency_bounds[0]) * np.random.rand()
                self.velocity[i] = self.inertia_weight * self.velocity[i] + (population[i] - best_solution) * frequency
                candidate = population[i] + self.velocity[i]
                candidate = np.clip(candidate, lb, ub)

                if np.random.rand() > self.pulse_rate[i]:
                    candidate = best_solution + self.levy_flight() * self.loudness[i]

                candidate_fitness = func(candidate)
                eval_count += 1

                if candidate_fitness < fitness[i]:
                    self.improvement_count[i] += 1
                    self.memory[i] = candidate
                    if candidate_fitness < self.personal_best_fitness[i]:
                        self.personal_best[i] = candidate
                        self.personal_best_fitness[i] = candidate_fitness

                if candidate_fitness < fitness[i] and np.random.rand() < self.loudness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    self.loudness[i] = min(1.0, self.loudness[i] * self.alpha)
                    if candidate_fitness < best_fitness:
                        self.pulse_rate[i] *= self.gamma

                if candidate_fitness < best_fitness:
                    best_solution = candidate
                    best_fitness = candidate_fitness

                if eval_count >= self.budget:
                    break

            improvement_rate = np.mean(self.improvement_count) / (self.stagnation_threshold + 1)
            self.dynamic_parameter_adjustment(improvement_rate)
            self.inertia_weight *= self.inertia_decay
            self.global_best_history.append(best_fitness)

            if len(self.global_best_history) > self.stagnation_threshold and self.global_best_history[-1] == self.global_best_history[-self.stagnation_threshold]:
                for j in range(self.population_size):
                    if np.random.rand() < 0.1:
                        population[j] = self.chaotic_injection(population[j], lb, ub)
                    if np.random.rand() < 0.3:
                        population[j] = self.quantum_tunneling(best_solution, lb, ub)

                self.loudness = np.random.uniform(0.5, 1.0, self.population_size)
                self.pulse_rate = np.random.uniform(0.2, 0.8, self.population_size)

        return best_solution