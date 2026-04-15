class HybridMetaheuristicOptimizer:
    ...
    def _adaptive_local_search(self, solution, func, bounds):
        step_size = self.local_search_intensity
        beta = 1.05  # Dynamic adjustment factor
        for _ in range(30):
            candidate = solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            if func(candidate) < func(solution):
                solution = candidate
                step_size *= beta
                beta += 0.01  # Dynamically adjust step size incrementally
            else:
                step_size *= 0.7
        return solution