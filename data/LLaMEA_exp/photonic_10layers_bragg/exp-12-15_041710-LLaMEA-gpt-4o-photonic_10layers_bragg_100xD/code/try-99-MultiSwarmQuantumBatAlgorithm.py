import numpy as np

class MultiSwarmQuantumBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.frequency_min = 0
        self.frequency_max = 3
        self.loudness = np.random.uniform(0.5, 1.0, (self.num_swarms, self.population_size))
        self.pulse_rate = np.random.uniform(0.2, 0.8, (self.num_swarms, self.population_size))
        self.alpha = 0.95
        self.gamma = 0.9
        self.beta = np.random.uniform(0, 1, (self.num_swarms, self.population_size))
        self.memory = np.zeros((self.num_swarms, self.population_size, self.dim))
        self.personal_best = np.random.uniform(0, 1, (self.num_swarms, self.population_size, self.dim))
        self.personal_best_fitness = np.full((self.num_swarms, self.population_size), np.inf)
        self.improvement_count = np.zeros((self.num_swarms, self.population_size))
        self.stagnation_threshold = 3
        self.global_best_history = []
        self.exploration_exploitation_tradeoff = 0.5
        self.adaptation_phase = 0.5
        self.chaos_factor = 0.1
        self.inertia_weight = 0.9
        self.inertia_decay = 0.985
        self.chaos_threshold = 0.3
        self.regroup_interval = 10  # Swarm regrouping interval

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

    def enhanced_quantum_perturbation(self, best_solution, lb, ub):
        omega = np.linspace(0.1, 1.0, self.dim)
        q = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0.1, 0.5, self.dim) * np.sin(omega * np.pi)
        return np.clip(best_solution + step_size * q * omega * (ub - lb), lb, ub)

    def adjust_population_size(self):
        if len(self.global_best_history) > self.stagnation_threshold:
            improvement = (self.global_best_history[-self.stagnation_threshold] - self.global_best_history[-1]) / self.global_best_history[-self.stagnation_threshold]
            if improvement < self.exploration_exploitation_tradeoff:
                self.population_size = max(int(self.initial_population_size / 2), 5)
            else:
                self.population_size = min(self.initial_population_size * 1.5, 50)

    def self_adaptive_parameters(self, improvement_rate):
        phase_adjustment = np.tanh(self.adaptation_phase * improvement_rate * 1.1)
        self.alpha = 0.85 + 0.15 * phase_adjustment
        for swarm in range(self.num_swarms):
            self.loudness[swarm] = np.clip(self.loudness[swarm] * (1.0 + phase_adjustment), 0.5, 1.0)
            self.pulse_rate[swarm] = np.clip(self.pulse_rate[swarm] * (1.0 - phase_adjustment), 0.2, 0.8)

    def regroup_swarms(self, lb, ub):
        new_population = np.random.uniform(lb, ub, (self.num_swarms, self.population_size, self.dim))
        for swarm in range(self.num_swarms):
            self.memory[swarm] = new_population[swarm]
            self.loudness[swarm] = np.random.uniform(0.5, 1.0, self.population_size)
            self.pulse_rate[swarm] = np.random.uniform(0.2, 0.8, self.population_size)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.num_swarms, self.population_size, self.dim))
        velocities = np.zeros((self.num_swarms, self.population_size, self.dim))
        fitness = np.array([[func(ind) for ind in swarm] for swarm in population])
        best_solution_idx = np.unravel_index(np.argmin(fitness), fitness.shape)
        best_solution = population[best_solution_idx]
        best_fitness = fitness.min()
        eval_count = self.num_swarms * self.population_size
        self.global_best_history.append(best_fitness)

        while eval_count < self.budget:
            for swarm in range(self.num_swarms):
                for i in range(self.population_size):
                    self.beta[swarm, i] = np.random.uniform(0, 1)
                    frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * self.beta[swarm, i]
                    velocities[swarm, i] *= self.inertia_weight
                    velocities[swarm, i] += (population[swarm, i] - best_solution) * frequency
                    candidate = population[swarm, i] + velocities[swarm, i]
                    candidate = np.clip(candidate, lb, ub)

                    if np.random.rand() > self.pulse_rate[swarm, i]:
                        candidate = best_solution + self.levy_flight() * self.loudness[swarm, i]

                    candidate_fitness = func(candidate)
                    eval_count += 1

                    if candidate_fitness < fitness[swarm, i]:
                        self.improvement_count[swarm, i] += 1
                        self.memory[swarm, i] = candidate
                        if candidate_fitness < self.personal_best_fitness[swarm, i]:
                            self.personal_best[swarm, i] = candidate
                            self.personal_best_fitness[swarm, i] = candidate_fitness

                    if candidate_fitness < fitness[swarm, i] and np.random.rand() < self.loudness[swarm, i]:
                        population[swarm, i] = candidate
                        fitness[swarm, i] = candidate_fitness
                        self.loudness[swarm, i] = min(1.0, self.loudness[swarm, i] * self.alpha)
                        if candidate_fitness < best_fitness:
                            self.pulse_rate[swarm, i] *= self.gamma

                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness

                    if eval_count >= self.budget:
                        break

                if eval_count < self.budget:
                    for i in range(self.population_size):
                        if np.random.rand() < 0.1:
                            opposite_candidate = self.opposition_based_learning(population[swarm, i], lb, ub)
                            opposite_candidate_fitness = func(opposite_candidate)
                            eval_count += 1

                            if opposite_candidate_fitness < fitness[swarm, i]:
                                population[swarm, i] = opposite_candidate
                                fitness[swarm, i] = opposite_candidate_fitness

                            if opposite_candidate_fitness < best_fitness:
                                best_solution = opposite_candidate
                                best_fitness = opposite_candidate_fitness

                            if eval_count >= self.budget:
                                break

            improvement_rate = np.mean(self.improvement_count) / (self.stagnation_threshold + 1)
            self.self_adaptive_parameters(improvement_rate)
            self.inertia_weight *= self.inertia_decay
            self.global_best_history.append(best_fitness)
            self.adjust_population_size()

            if eval_count % self.regroup_interval == 0:
                self.regroup_swarms(lb, ub)

            if len(self.global_best_history) > self.stagnation_threshold and self.global_best_history[-1] == self.global_best_history[-self.stagnation_threshold]:
                for swarm in range(self.num_swarms):
                    for j in range(self.population_size):
                        if np.random.rand() < self.chaos_threshold:
                            population[swarm, j] = self.chaotic_perturbation(population[swarm, j], lb, ub, self.chaos_factor)
                        if np.random.rand() < 0.35:
                            population[swarm, j] = self.enhanced_quantum_perturbation(best_solution, lb, ub)
                
                    self.loudness[swarm] = np.random.uniform(0.5, 1.0, self.population_size)
                    self.pulse_rate[swarm] = np.random.uniform(0.2, 0.8, self.population_size)

        return best_solution