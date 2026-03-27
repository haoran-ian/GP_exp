import numpy as np

class AdaptiveAnnealedCuckoo:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.nests = 25  # number of nests (solutions)
        self.pa = 0.3   # adjusted probability of abandoning a solution
        self.beta = 1.5  # Levy flight parameter
        self.lr = 0.8    # Local refinement probability
        self.pop = None
        self.fitness = None

    def levy_flight(self, size, scale=1.0):
        sigma = (np.math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                 (np.math.gamma((1 + self.beta) / 2) * self.beta *
                  2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, size=size) * scale
        v = np.random.normal(0, 1, size=size)
        step = u / np.abs(v) ** (1 / self.beta)
        return step

    def differential_evolution(self, target_idx, F=0.5, CR=0.8):
        idxs = [idx for idx in range(self.nests) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = self.pop[a] + F * (self.pop[b] - self.pop[c])
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, self.pop[target_idx])
        return np.clip(trial, self.lb, self.ub)

    def simulated_annealing(self, candidate, current_fitness, temperature):
        new_candidate = candidate + np.random.normal(0, 0.1, self.dim) * temperature
        new_candidate = np.clip(new_candidate, self.lb, self.ub)
        new_fitness = func(new_candidate)
        if new_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness) / (temperature + 1e-9)):
            return new_candidate, new_fitness
        return candidate, current_fitness

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.pop = np.random.uniform(self.lb, self.ub, (self.nests, self.dim))
        self.fitness = np.array([func(ind) for ind in self.pop])
        best_idx = np.argmin(self.fitness)
        best_solution = self.pop[best_idx]
        
        for t in range(self.budget - self.nests):
            new_pop = self.pop.copy()
            temperature = 1 - t / self.budget  # simulated annealing temperature
            for i in range(self.nests):
                exploration_scale = 1 + (t / self.budget)
                mutation_rate = np.random.rand() * (0.2 + 0.8 * np.exp(-t/self.budget))  # Adaptive mutation rate
                
                if np.random.rand() < self.lr:
                    candidate = self.differential_evolution(i, F=np.random.rand() * mutation_rate, CR=np.random.rand())
                else:
                    step_size = self.levy_flight(self.dim, scale=exploration_scale) * (self.pop[i] - best_solution)
                    candidate = self.pop[i] + step_size * np.random.uniform(-1, 1.2, self.dim)
                
                candidate = np.clip(candidate, self.lb, self.ub)
                f_candidate = func(candidate)
                candidate, f_candidate = self.simulated_annealing(candidate, f_candidate, temperature)
                
                if f_candidate < self.fitness[i]:
                    new_pop[i] = candidate
                    self.fitness[i] = f_candidate

            # Abandon some nests
            diversity_factor = np.std(self.fitness) / (np.mean(self.fitness) + 1e-9)
            self.pa = 0.1 + 0.2 * (1 - diversity_factor)  # Dynamic pa value inversely proportional to diversity
            abandon = np.random.rand(self.nests) < self.pa
            for i in range(self.nests):
                if abandon[i] and t < self.budget - self.nests:
                    new_pop[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    self.fitness[i] = func(new_pop[i])
            
            self.pop = new_pop
            best_idx = np.argmin(self.fitness)
            best_solution = self.pop[best_idx]

        return best_solution