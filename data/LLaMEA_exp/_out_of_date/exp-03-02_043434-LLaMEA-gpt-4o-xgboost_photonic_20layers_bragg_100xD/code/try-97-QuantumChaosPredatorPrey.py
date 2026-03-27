import numpy as np

class QuantumChaosPredatorPrey:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.population = np.random.rand(self.population_size, dim)
        self.initial_population_size = self.population_size
        self.F_base = 0.5
        self.CR_base = 0.9
        self.inertia_weight = 0.9

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def chaotic_local_search(self, position, lb, ub, chaos_level=0.1):
        # Changed to dynamic chaos level
        dynamic_chaos_level = chaos_level * (1 + np.random.rand())
        chaotic_step = dynamic_chaos_level * (np.random.rand(self.dim) - 0.5) * (ub - lb)
        new_position = position + chaotic_step
        return np.clip(new_position, lb, ub)

    def quantum_tunneling(self, position, best_position, lb, ub):
        tunneling_step = np.random.normal(0, 0.1, size=self.dim) * (best_position - position)
        new_position = position + tunneling_step
        return np.clip(new_position, lb, ub)

    def predator_prey_dynamics(self, func, lb, ub, best_solution, predator_influence=0.05):
        # Changed to adaptive predator influence
        adaptive_predator_influence = predator_influence * (1.0 + 0.1 * np.random.rand())
        predators = self.population[np.random.choice(self.population.shape[0], 2, replace=False)]
        prey = self.population[np.random.choice(self.population.shape[0], 1, replace=False)][0]
        
        for predator in predators:
            influence = adaptive_predator_influence * (predator - prey)
            prey = np.clip(prey + influence, lb, ub)
        
        if func(prey) < func(best_solution):
            best_solution = prey

        return best_solution

    def differential_evolution(self, func, lb, ub):
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)

        while evaluations < self.budget:
            self.population_size = max(int(self.initial_population_size * (1 - evaluations / self.budget)), 4)
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                F_dynamic = self.F_base + self.inertia_weight * np.random.rand()
                CR_dynamic = self.CR_base - self.inertia_weight * np.random.rand()

                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                if np.random.rand() < 0.5:
                    trial_vector += self.levy_flight(1.5) * (trial_vector - self.population[i])

                trial_vector = self.chaotic_local_search(trial_vector, lb, ub)

                # Added: increased stochastic diversity
                if np.random.rand() < 0.1:
                    trial_vector = lb + np.random.rand(self.dim) * (ub - lb)

                trial_vector = self.quantum_tunneling(trial_vector, best_solution if best_solution is not None else trial_vector, lb, ub)

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

            best_solution = self.predator_prey_dynamics(func, lb, ub, best_solution)

        return best_solution, best_fitness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_fitness = self.differential_evolution(func, lb, ub)
        return best_solution