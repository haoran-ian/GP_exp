import numpy as np

class ImprovedEMADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.best_solution = None
        self.best_fitness = np.inf
        self.beta = 1.5  # Parameter for Lévy flight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_size = self.initial_population_size
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = pop_size

        while evals < self.budget:
            F = 0.5 + 0.5 * np.random.rand()  # Dynamic F
            CR = 0.3 + 0.7 * np.random.rand()  # Dynamic CR

            for i in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])

                trial_fitness = func(trial)
                evals += 1
                if evals >= self.budget:
                    break

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

            # Stochastic Lévy flight mutation for exploration
            if np.random.rand() < 0.1:
                flight = self.levy_flight(lb, ub)
                candidate = np.clip(self.best_solution + flight * (ub - lb), lb, ub)
                candidate_fitness = func(candidate)
                evals += 1
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate

            # Dynamic Population Sizing
            if evals / self.budget > 0.5:
                pop_size = max(10, int(self.initial_population_size * (1 - (evals / self.budget))))

        return self.best_solution, self.best_fitness

    def levy_flight(self, lb, ub):
        sigma_u = (np.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                   (np.gamma((1 + self.beta) / 2) * self.beta * 2**((self.beta - 1) / 2)))**(1 / self.beta)
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / self.beta)
        return step * 0.01 * (ub - lb)