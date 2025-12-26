import numpy as np

class EnhancedHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * self.dim  # Slightly larger initial population
        self.crossover_rate = 0.85  # Adjust crossover to increase diversity
        self.scaling_factor = 0.75  # Lower scaling factor for more stable convergence
        self.temperature = 100.0
        self.cooling_rate = 0.995  # Fine-tuned cooling for temperature
        self.inertia_weight = 0.9  # Adaptive inertia weight based on budget usage
        self.cognitive_constant = 1.6  # Increased for better exploitation
        self.social_constant = 1.4  # Reduced to limit premature convergence

    def levy_flight(self, L):
        u = np.random.normal(0, 1, self.dim) * L
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1.0 / 3.0)
        return step

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.zeros_like(population)
        personal_best = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        eval_count = self.population_size

        while eval_count < self.budget:
            mutation_prob = max(0.2, min(0.4, 1.0 - (eval_count / self.budget)))  # Enhanced adaptive mutation probability
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                # Update velocities using PSO dynamics
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                adaptive_inertia = self.inertia_weight * (1 - eval_count / self.budget)
                velocities[i] = (
                    adaptive_inertia * velocities[i]
                    + self.cognitive_constant * r1 * (personal_best[i] - population[i])
                    + self.social_constant * r2 * (global_best - population[i])
                )

                # Update position
                population[i] += velocities[i]

                # Levy flight perturbation
                levy_step = self.levy_flight(0.1 + 0.05 * (eval_count / self.budget))  # More gradual adjustment
                if np.random.rand() < mutation_prob:
                    population[i] += levy_step

                # Boundary constraints
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])

                # Evaluate fitness
                fitness = func(population[i])
                eval_count += 1

                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness
                    if fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = fitness

                        # Adjust cooling dynamically
                        self.cooling_rate *= 0.9925  # Adjust cooling to stabilize

        return global_best, global_best_fitness