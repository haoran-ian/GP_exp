import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F_base = 0.5  # Base differential weight
        self.CR = 0.9  # Crossover probability
        self.greedy_probability = 0.1
        self.elitism_rate = 0.2
        self.init_population()

    def init_population(self):
        self.population = np.random.rand(self.population_size, self.dim)
        self.scores = np.full(self.population_size, np.inf)
    
    def adapt_scaling_factor(self, iteration, max_iterations):
        # Adaptive scaling factor that decreases over time to encourage exploration initially and exploitation later
        return self.F_base * (1 - iteration / max_iterations)

    def mutate(self, target_idx, F):
        indices = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        donor_vector = self.population[a] + F * (self.population[b] - self.population[c])
        return np.clip(donor_vector, 0, 1)

    def crossover(self, target, donor):
        trial = np.copy(target)
        crossover_points = np.random.rand(self.dim) < self.CR
        trial[crossover_points] = donor[crossover_points]
        return trial

    def local_greedy_search(self, individual):
        perturbation = np.random.uniform(-0.1, 0.1, self.dim)
        greedy_candidate = np.clip(individual + perturbation, 0, 1)
        return greedy_candidate

    def __call__(self, func):
        bounds = func.bounds
        max_iterations = self.budget // self.population_size

        for i in range(self.population_size):
            self.population[i] = bounds.lb + (bounds.ub - bounds.lb) * self.population[i]
            self.scores[i] = func(self.population[i])
            self.budget -= 1

        iteration = 0
        while self.budget > 0:
            iteration += 1
            F = self.adapt_scaling_factor(iteration, max_iterations)

            # Elitism: Keep a portion of the best solutions untouched
            num_elites = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.scores)[:num_elites]
            non_elite_indices = [i for i in range(self.population_size) if i not in elite_indices]

            for i in non_elite_indices:
                donor = self.mutate(i, F)
                trial = self.crossover(self.population[i], donor)

                if np.random.rand() < self.greedy_probability:
                    trial = self.local_greedy_search(trial)

                trial_denormalized = bounds.lb + (bounds.ub - bounds.lb) * trial
                trial_score = func(trial_denormalized)
                self.budget -= 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial
                    self.scores[i] = trial_score

                if self.budget <= 0:
                    break

        best_idx = np.argmin(self.scores)
        best_solution = self.population[best_idx]
        return bounds.lb + (bounds.ub - bounds.lb) * best_solution