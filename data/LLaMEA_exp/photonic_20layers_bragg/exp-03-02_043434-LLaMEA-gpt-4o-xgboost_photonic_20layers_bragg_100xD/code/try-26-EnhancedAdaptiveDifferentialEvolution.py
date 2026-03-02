import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim  # Adaptive population size
        self.population = np.random.rand(self.population_size, dim)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (np.abs(v) ** (1 / L))
        return step

    def adaptive_chaotic_map(self, chaos_level=0.1):
        # Incorporate logistic map for chaotic behavior
        r = 4.0  # Logistic map parameter for chaos
        x = np.random.rand(self.dim)
        chaotic_sequence = r * x * (1 - x)
        return chaos_level * (chaotic_sequence - 0.5)

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Introduce dynamic adaptation for F and CR
                F_dynamic = 0.5 + 0.3 * np.random.rand()
                CR_dynamic = 0.9 - 0.2 * np.random.rand()
                
                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                
                if np.random.rand() < 0.5:  # Incorporate multi-angle Lévy flights
                    angle = np.random.rand() * 2 * np.pi
                    levy_step = self.levy_flight(1.5)
                    trial_vector += np.array([levy_step * np.cos(angle), levy_step * np.sin(angle)])

                # Apply adaptive chaotic map to trial vector
                trial_vector += self.adaptive_chaotic_map()
                trial_vector = np.clip(trial_vector, lb, ub)

                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_vector

                if evaluations >= self.budget:
                    break

        return best_solution, best_fitness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_fitness = self.differential_evolution(func, lb, ub)
        return best_solution