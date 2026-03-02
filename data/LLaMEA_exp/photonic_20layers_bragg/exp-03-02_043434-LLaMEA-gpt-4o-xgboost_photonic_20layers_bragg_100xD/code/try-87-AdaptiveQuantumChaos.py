import numpy as np

class AdaptiveQuantumChaos:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20 * dim
        self.population_size = self.initial_population_size
        self.population = np.random.rand(self.population_size, dim)
        self.F_base = 0.5
        self.CR_base = 0.9
        self.memory_size = 5
        self.memory = []

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def chaotic_local_search(self, position, lb, ub, chaos_level=0.1):
        dynamic_chaos_level = chaos_level * (1 + np.random.rand())
        chaotic_step = dynamic_chaos_level * (np.random.rand(self.dim) - 0.5) * (ub - lb)
        new_position = position + chaotic_step
        return np.clip(new_position, lb, ub)

    def quantum_tunneling(self, position, best_position, lb, ub):
        tunneling_step = np.random.normal(0, 0.1, size=self.dim) * (best_position - position)
        new_position = position + tunneling_step
        return np.clip(new_position, lb, ub)
    
    def update_memory(self, candidate, fitness):
        self.memory.append((candidate, fitness))
        if len(self.memory) > self.memory_size:
            self.memory.sort(key=lambda x: x[1])
            self.memory = self.memory[:self.memory_size]

    def calculate_diversity(self):
        return np.mean(np.std(self.population, axis=0))

    def differential_evolution(self, func, lb, ub):
        best_solution = None
        best_fitness = np.inf
        evaluations = 0
        self.population = lb + (ub - lb) * self.population
        fitness = np.apply_along_axis(func, 1, self.population)

        while evaluations < self.budget:
            diversity = self.calculate_diversity()
            self.population_size = max(int(self.initial_population_size * (0.5 + diversity)), 4)
            
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                F_dynamic = self.F_base + diversity * np.random.rand()
                CR_dynamic = self.CR_base - diversity * np.random.rand()

                mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                if np.random.rand() < 0.5:
                    trial_vector += self.levy_flight(1.5) * (trial_vector - self.population[i])
                
                trial_vector = self.chaotic_local_search(trial_vector, lb, ub)
                trial_vector = self.quantum_tunneling(trial_vector, best_solution if best_solution is not None else trial_vector, lb, ub)

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial_vector
                    fitness[i] = trial_fitness

                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial_vector
                
                self.update_memory(trial_vector, trial_fitness)

                if evaluations >= self.budget:
                    break
            
        return best_solution, best_fitness

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_fitness = self.differential_evolution(func, lb, ub)
        return best_solution