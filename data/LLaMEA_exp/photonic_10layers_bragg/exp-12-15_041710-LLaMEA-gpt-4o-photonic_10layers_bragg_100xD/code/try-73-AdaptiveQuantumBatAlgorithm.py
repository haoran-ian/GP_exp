import numpy as np

class AdaptiveQuantumBatAlgorithm:
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
        self.memory = np.zeros((self.population_size, self.dim))
        self.personal_best = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.improvement_count = np.zeros(self.population_size)
        self.stagnation_threshold = 3
        self.global_best_history = []
        self.exploration_exploitation_tradeoff = 0.5
        self.adaptation_phase = 0.5
        self.chaos_factor = 0.1
        self.inertia_weight = 0.9
        self.inertia_decay = 0.98
        self.chaos_threshold = 0.3
        self.fitness_diversity_threshold = 0.05

    def opposition_based_learning(self, candidate, lb, ub):
        return lb + ub - candidate

    def levy_flight(self, lam=1.5):
        sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2) / 
                 (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)))**(1 / lam)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / lam)
        return step

    def chaotic_perturbation(self, solution, lb, ub, chaos_intensity):
        chaotic_factor = np.random.uniform(-chaos_intensity, chaos_intensity, self.dim)
        perturbed_solution = solution + chaotic_factor * (ub - lb)
        return np.clip(perturbed_solution, lb, ub)

    def quantum_harmonic_oscillation(self, best_solution, lb, ub, diversity_factor):
        omega = np.linspace(0.1, 1.0, self.dim)
        q = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0, 1, self.dim) * np.sin(omega * np.pi)
        return np.clip(best_solution + step_size * q * omega * (ub - lb) * diversity_factor, lb, ub)

    def adjust_population_size(self):
        if len(self.global_best_history) > self.stagnation_threshold:
            improvement = (self.global_best_history[-self.stagnation_threshold] - self.global_best_history[-1]) / self.global_best_history[-self.stagnation_threshold]
            if improvement < self.exploration_exploitation_tradeoff:
                self.population_size = max(int(self.initial_population_size / 2), 5)
            else:
                self.population_size = min(self.initial_population_size * 1.5, 50)

    def self_adaptive_parameters(self, improvement_rate, fitness_diversity):
        phase_adjustment = np.tanh(self.adaptation_phase * improvement_rate * 1.1)
        self.alpha = 0.85 + 0.15 * phase_adjustment
        self.loudness = np.clip(self.loudness * (1.0 + phase_adjustment), 0.5, 1.0)
        self.pulse_rate = np.clip(self.pulse_rate * (1.0 - phase_adjustment), 0.2, 0.8)

        if fitness_diversity < self.fitness_diversity_threshold:
            self.loudness *= 1.1
            self.pulse_rate *= 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        eval_count = self.population_size
        self.global_best_history.append(best_fitness)

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

            if eval_count < self.budget:
                for i in range(self.population_size):
                    if np.random.rand() < 0.1:
                        opposite_candidate = self.opposition_based_learning(population[i], lb, ub)
                        opposite_candidate_fitness = func(opposite_candidate)
                        eval_count += 1

                        if opposite_candidate_fitness < fitness[i]:
                            population[i] = opposite_candidate
                            fitness[i] = opposite_candidate_fitness

                        if opposite_candidate_fitness < best_fitness:
                            best_solution = opposite_candidate
                            best_fitness = opposite_candidate_fitness

                        if eval_count >= self.budget:
                            break

            improvement_rate = np.mean(self.improvement_count) / (self.stagnation_threshold + 1)
            fitness_diversity = np.std(fitness) / (np.mean(fitness) + 1e-10)
            self.self_adaptive_parameters(improvement_rate, fitness_diversity)
            self.inertia_weight *= self.inertia_decay
            self.global_best_history.append(best_fitness)
            self.adjust_population_size()

            if len(self.global_best_history) > self.stagnation_threshold and self.global_best_history[-1] == self.global_best_history[-self.stagnation_threshold]:
                diversity_factor = 1 + fitness_diversity * 0.1
                for j in range(self.population_size):
                    if np.random.rand() < self.chaos_threshold:
                        population[j] = self.chaotic_perturbation(population[j], lb, ub, self.chaos_factor)
                    if np.random.rand() < 0.35:
                        population[j] = self.quantum_harmonic_oscillation(best_solution, lb, ub, diversity_factor)
                
                self.loudness = np.random.uniform(0.5, 1.0, self.population_size)
                self.pulse_rate = np.random.uniform(0.2, 0.8, self.population_size)

        return best_solution