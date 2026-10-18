"""
tests/test_allocator.py
Unit tests for the Respond resource allocation engine.
Verifies greedy marginal benefit optimization against brute-force optimal allocation on small cases
per BUILD.md Section 5 and Section 9.4.
"""

import unittest
import itertools
import numpy as np


def compute_benefit(teams_assigned, exposure, r=0.30):
    """
    Compute public health benefit of allocating teams to a ward.
    Formula: Benefit(n) = exposure * (1 - (1 - r)^n).
    
    Inputs:
        teams_assigned (int): Number of teams deployed to this ward.
        exposure (float): Risk-weighted population exposure.
        r (float): Diminishing return factor per team (default 0.30).
    Outputs:
        float: Cumulative intervention benefit.
    """
    return exposure * (1.0 - (1.0 - r) ** teams_assigned)


def greedy_allocate(exposures, total_teams, r=0.30):
    """
    Greedy marginal benefit allocation of discrete teams across wards.
    
    Inputs:
        exposures (list of float): Exposures per ward.
        total_teams (int): Total available inspection teams.
        r (float): Effectiveness factor.
    Outputs:
        list of int: Teams allocated to each ward.
    """
    n_wards = len(exposures)
    allocation = [0] * n_wards
    for _ in range(total_teams):
        marginal_gains = []
        for i in range(n_wards):
            curr_b = compute_benefit(allocation[i], exposures[i], r)
            next_b = compute_benefit(allocation[i] + 1, exposures[i], r)
            marginal_gains.append(next_b - curr_b)
        best_ward = int(np.argmax(marginal_gains))
        allocation[best_ward] += 1
    return allocation


def brute_force_allocate(exposures, total_teams, r=0.30):
    """
    Exhaustive search across all integer partitions of teams for exact global optimum.
    
    Inputs:
        exposures (list of float): Exposures per ward.
        total_teams (int): Total available teams.
        r (float): Effectiveness factor.
    Outputs:
        tuple: (best_allocation, max_total_benefit)
    """
    n_wards = len(exposures)
    best_alloc = None
    best_benefit = -1.0

    def generate_partitions(n, k):
        if k == 1:
            yield (n,)
        else:
            for i in range(n + 1):
                for p in generate_partitions(n - i, k - 1):
                    yield (i,) + p

    for partition in generate_partitions(total_teams, n_wards):
        total_b = sum(compute_benefit(partition[i], exposures[i], r) for i in range(n_wards))
        if total_b > best_benefit:
            best_benefit = total_b
            best_alloc = list(partition)

    return best_alloc, best_benefit


class TestAllocator(unittest.TestCase):
    """
    Validation test suite comparing greedy marginal team allocation to brute force search.
    """

    def test_greedy_vs_brute_force_small_case(self):
        """Test that greedy allocation matches exact combinatorial optimum on 3 wards and 5 teams."""
        exposures = [150000.0, 95000.0, 40000.0]
        total_teams = 5
        r = 0.30

        greedy_res = greedy_allocate(exposures, total_teams, r)
        brute_res, brute_val = brute_force_allocate(exposures, total_teams, r)

        greedy_val = sum(compute_benefit(greedy_res[i], exposures[i], r) for i in range(len(exposures)))
        
        # Because the benefit function is strictly concave with diminishing marginal returns,
        # greedy allocation is mathematically guaranteed to equal the global optimum.
        self.assertEqual(greedy_res, brute_res)
        self.assertAlmostEqual(greedy_val, brute_val, places=4)

    def test_zero_teams(self):
        """Test allocation when zero teams are available."""
        exposures = [100.0, 200.0]
        res = greedy_allocate(exposures, 0)
        self.assertEqual(res, [0, 0])


if __name__ == "__main__":
    unittest.main()
