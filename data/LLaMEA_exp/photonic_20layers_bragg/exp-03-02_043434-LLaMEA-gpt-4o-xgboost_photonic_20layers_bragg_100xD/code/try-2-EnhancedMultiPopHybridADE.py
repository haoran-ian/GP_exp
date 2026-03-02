import numpy as np

class EnhancedMultiPopHybridADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim  # Adaptive population size
        self.num_subpopulations = 3
        self.subpopulations = [np.random.rand(self.population_size // self.num_subpopulations, dim) for _ in range(self.num_subpopulations)]
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.info_sharing_interval = max(1, budget // 5)  # Share info periodically

    def levy_flight(self, L):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / L)
        return step

    def differential_evolution(self, func, lb, ub):
        bounds = np.array([lb, ub])
        best_solution = None
        best_fitness = np.inf
        evaluations = 0

        for subpop in self.subpopulations:
            subpop[:] = lb + (ub - lb) * subpop

        fitness = [np.apply_along_axis(func, 1, subpop) for subpop in self.subpopulations]
        
        while evaluations < self.budget:
            for sp_index, subpop in enumerate(self.subpopulations):
                for i in range(subpop.shape[0]):
                    indices = [idx for idx in range(subpop.shape[0]) if idx != i]
                    a, b, c = subpop[np.random.choice(indices, 3, replace=False)]

                    # Dynamic adaptation for F and CR
                    F_dynamic = 0.5 + 0.3 * np.random.rand()
                    CR_dynamic = 0.9 - 0.2 * np.random.rand()
                    
                    mutant_vector = np.clip(a + F_dynamic * (b - c), lb, ub)
                    crossover_mask = np.random.rand(self.dim) < CR_dynamic
                    trial_vector = np.where(crossover_mask, mutant_vector, subpop[i])
                    
                    if np.random.rand() < 0.5:  # Incorporate Lévy flights
                        trial_vector += self.levy_flight(1.5) * (trial_vector - subpop[i])
                    
                    trial_fitness = func(trial_vector)
                    evaluations += 1
                    
                    if trial_fitness < fitness[sp_index][i]:
                        subpop[i] = trial_vector
                        fitness[sp_index][i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial_vector

                    if evaluations >= self.budget:
                        break

            # Periodic information sharing between subpopulations
            if evaluations % self.info_sharing_interval == 0:
                self.share_info_between_subpops()

        return best_solution, best_fitness

    def share_info_between_subpops(self):
        best_individuals = [subpop[np.argmin(fit)] for subpop, fit in zip(self.subpopulations, fitness)]
        for subpop in self.subpopulations:
            for i in range(subpop.shape[0]):
                if np.random.rand() < 0.1:  # Small chance to replace an individual
                    subpop[i] = best_individuals[np.random.choice(len(best_individuals))]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution, best_fitness = self.differential_evolution(func, lb, ub)
        return best_solution