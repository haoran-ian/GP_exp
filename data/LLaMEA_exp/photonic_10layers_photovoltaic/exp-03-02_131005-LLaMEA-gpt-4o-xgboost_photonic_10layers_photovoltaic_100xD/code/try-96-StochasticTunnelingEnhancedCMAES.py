import numpy as np

class StochasticTunnelingEnhancedCMAES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.memory = []
        self.alpha = 0.5  # Stochastic tunneling parameter

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = int(np.sqrt(self.budget))
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        fitness = np.array([self._evaluate(func, ind) for ind in population])

        weights = np.log(population_size + 0.5) - np.log(np.arange(1, population_size + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_s = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_1 = 2 / ((self.dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2)**2 + mu_eff))
        damping = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_s

        sigma = 0.3 * (ub - lb)
        mean = np.mean(population, axis=0)
        cov_matrix = np.eye(self.dim)
        p_c = np.zeros(self.dim)
        p_s = np.zeros(self.dim)

        while self.evaluations < self.budget:
            cov_matrix = (1 - c_1 - c_mu) * cov_matrix + c_1 * np.outer(p_c, p_c)
            values, vectors = np.linalg.eigh(cov_matrix)
            B = vectors
            D = np.diag(np.sqrt(values))
            C = np.dot(np.dot(B, D), B.T)

            candidates = [mean + sigma * np.dot(D, np.random.randn(self.dim)) for _ in range(population_size)]
            candidates = np.clip(candidates, lb, ub)
            candidate_fitness = np.array([self._evaluate(func, ind) for ind in candidates])

            for i in range(population_size):
                candidate_fitness[i] = -np.exp(-self.alpha * candidate_fitness[i])

            idx = np.argsort(candidate_fitness)
            best_candidates = candidates[idx[:len(weights)]]
            mean_new = np.dot(weights, best_candidates)

            p_s = (1 - c_s) * p_s + np.sqrt(c_s * (2 - c_s) * mu_eff) * np.dot(B, np.linalg.solve(D, mean_new - mean) / sigma)
            sigma *= np.exp((c_s / damping) * (np.linalg.norm(p_s) / np.sqrt(1 - (1 - c_s)**(2 * self.budget / population_size)) - 1))

            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean_new - mean) / sigma

            mean = mean_new

        best_idx = np.argmin(fitness)
        return population[best_idx]

    def _evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            raise RuntimeError("Exceeded budget")
        self.evaluations += 1
        return func(individual)