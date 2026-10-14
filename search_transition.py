"""Randomly search periodic degree patterns for numerical phase transitions.

For f_d(r) = lambda / (1+r)^d, compose the maps in the order given
by a degree pattern. Odd primitive periods: test period doubling of F.
Even primitive periods: test for a saddle-node of F.

Requires: numpy, scipy, joblib
"""

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import csv
import time
import numpy as np
from scipy.optimize import brentq, least_squares
from joblib import Parallel, delayed, parallel_config

# -------------------------- Settings --------------------------
PERIOD_LENGTH = 10
N_SAMPLES = 15_000
DEGREE_MIN, DEGREE_MAX = 2, 100
SEED = 1234
N_JOBS = -1  # all available CPUs

LAMBDA_MIN = 1e-10
INITIAL_LAMBDA_MAX = 1000.0
MAX_LAMBDA = 1e30
EXPANSION_FACTOR = 10.0
SCAN_POINTS = 200
R_GRID_SIZE = 1200
JUMP_TOLERANCE = 1e-3
EXPONENT_TOLERANCE = 0.1

OUTPUT_FILE = "transition_results.csv"
INTERESTING_FILE = "interesting_patterns.csv"


# ----------------------- Basic functions ----------------------
def minimal_period(pattern):
    """Turn (2, 3, 2, 3) into (2, 3)."""
    pattern = tuple(map(int, pattern))
    if not pattern or min(pattern) <= 0:
        raise ValueError("Degree patterns must contain positive integers.")
    for k in range(1, len(pattern) + 1):
        if len(pattern) % k == 0 and pattern[:k] * (len(pattern) // k) == pattern:
            return pattern[:k]


def occupation(r):
    return r / (1 + r)


def map_and_slope(pattern, lam, r):
    """Compute F(r) and F'(r), using logarithms to avoid huge powers."""
    y = np.asarray(r, dtype=float)
    slope = np.ones_like(y)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for degree in reversed(pattern):
            next_y = np.exp(np.log(lam) - degree * np.log1p(y))
            slope *= -degree * next_y / (1 + y)
            y = next_y
    return y, slope


def F(pattern, lam, r):
    return map_and_slope(pattern, lam, r)[0]


def dF(pattern, lam, r):
    return map_and_slope(pattern, lam, r)[1]


def fixed_points(pattern, lam, grid_size=R_GRID_SIZE):
    """Find roots of F(r) = r detected by sign changes on a grid."""
    linear = np.linspace(0, lam, grid_size)
    log_grid = np.geomspace(max(lam * 1e-12, 1e-15), lam, max(150, grid_size // 2))
    grid = np.unique(np.r_[linear, log_grid])
    values = F(pattern, lam, grid) - grid
    with np.errstate(invalid="ignore", over="ignore"):
        crossings = np.flatnonzero(values[:-1] * values[1:] < 0)
    roots = sorted(brentq(lambda r: float(F(pattern, lam, r) - r),
                          grid[i], grid[i + 1], xtol=1e-12, rtol=1e-12)
                   for i in crossings)
    unique = []
    for root in roots:
        if not unique or root - unique[-1] > 1e-8 * max(1, lam):
            unique.append(root)
    return unique


def lambda_windows():
    """Scan 1e-10..100, then 100..1000, etc. up to MAX_LAMBDA."""
    low = LAMBDA_MIN
    high = min(INITIAL_LAMBDA_MAX, MAX_LAMBDA)
    while low < high:
        yield np.geomspace(low, high, SCAN_POINTS), high
        low, high = high, min(high * EXPANSION_FACTOR, MAX_LAMBDA)


# --------------------- Odd-period case ------------------------
def odd_fixed_point(pattern, lam):
    """Odd compositions are decreasing, so their fixed point is unique."""
    return brentq(lambda r: float(F(pattern, lam, r) - r),
                  0, lam, xtol=1e-13, rtol=1e-13)


def odd_critical_point(pattern):
    """Find F(r_c) = r_c and F'(r_c) = -1."""
    def condition(lam):
        return float(dF(pattern, lam, odd_fixed_point(pattern, lam)) + 1)

    for lambdas, search_max in lambda_windows():
        values = [condition(lam) for lam in lambdas]
        for i in range(len(lambdas) - 1):
            if abs(values[i]) < 1e-14:
                critical = float(lambdas[i])
            elif values[i] * values[i + 1] < 0:
                critical = brentq(condition, lambdas[i], lambdas[i + 1],
                                  xtol=1e-13, rtol=1e-13)
            else:
                continue
            return critical, odd_fixed_point(pattern, critical), search_max
    return None


def two_cycle_branches(pattern, lam):
    """Return lower and upper nontrivial roots of F(F(r)) = r."""
    middle = odd_fixed_point(pattern, lam)
    g = lambda r: float(F(pattern, lam, F(pattern, lam, r)) - r)
    offsets = max(1.0, middle) * np.geomspace(1e-7, 0.9, 140)
    left = right = None

    for offset in offsets:
        x = middle - offset
        if x <= 0:
            break
        if g(x) < -1e-13:
            left = x
            break
    for offset in offsets:
        x = middle + offset
        if x >= lam:
            break
        if g(x) > 1e-13:
            right = x
            break

    if left is None or right is None or g(0) * g(left) >= 0 or g(right) * g(lam) >= 0:
        return None
    lower = brentq(g, 0, left, xtol=1e-13, rtol=1e-13)
    upper = brentq(g, right, lam, xtol=1e-13, rtol=1e-13)
    return lower, upper


def odd_pitchfork_test(pattern, critical, r_c):
    """Check gap ~ epsilon^1/2 with negligible intercept near lambda_c."""
    eps_values = (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4,
                  5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6)
    measurements = []
    for eps in eps_values:
        branches = two_cycle_branches(pattern, critical * (1 + eps))
        if branches is not None:
            lower, upper = branches
            gap = occupation(upper) - occupation(lower)
            if np.isfinite(gap) and gap > 0:
                measurements.append((eps, gap))

    if len(measurements) < 5:
        return False, None, None

    # Fit using the closest half of the available observations.
    points = np.array(sorted(measurements)[:max(5, len(measurements) // 2)])
    eps, gaps = points[:, 0], points[:, 1]
    _, jump = np.polyfit(np.sqrt(eps), gaps, 1)
    beta, _ = np.polyfit(np.log(eps), np.log(gaps), 1)

    below = critical * (1 - 1e-4)
    above = critical * (1 + 1e-4)
    crosses = (dF(pattern, below, odd_fixed_point(pattern, below)) > -1
               and dF(pattern, above, odd_fixed_point(pattern, above)) < -1)
    is_second_order = (abs(jump) < JUMP_TOLERANCE
                       and abs(beta - 0.5) < EXPONENT_TOLERANCE and crosses)
    return bool(is_second_order), float(jump), float(beta)


# --------------------- Even-period case -----------------------
def even_critical_point(pattern):
    """Find the saddle-node F(r_c)=r_c, F'(r_c)=1."""
    guess = None
    for lambdas, search_max in lambda_windows():
        for lam in lambdas:
            roots = fixed_points(pattern, lam)
            if len(roots) >= 3:
                i = int(np.argmin(np.diff(roots)))
                guess = (float(lam), (roots[i] + roots[i + 1]) / 2)
                break
        if guess is not None:
            break
    if guess is None:
        return None

    def equations(log_coords):
        lam, r = np.exp(log_coords)
        value, slope = map_and_slope(pattern, lam, r)
        return [(float(value) - r) / max(1, r), float(slope) - 1]

    solution = least_squares(
        equations, np.log(guess),
        bounds=(np.log([max(LAMBDA_MIN, 1e-15), 1e-15]),
                np.log([MAX_LAMBDA, MAX_LAMBDA])),
        xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=1000)
    if not solution.success or np.linalg.norm(solution.fun) > 1e-8:
        return None
    critical, saddle = np.exp(solution.x)
    return float(critical), float(saddle), search_max


def even_jump(pattern, critical, saddle):
    """Compare occupation at saddle-node and surviving fixed point."""
    surviving = brentq(lambda r: float(F(pattern, critical, r) - r),
                       0, critical, xtol=1e-12, rtol=1e-12)
    return float(abs(occupation(surviving) - occupation(saddle)))


# --------------------- Pattern analysis -----------------------
def analyze_one(pattern):
    original = tuple(map(int, pattern))
    pattern = minimal_period(original)
    result = dict(input_pattern=original, primitive_pattern=pattern,
                  period=len(pattern), status="ok", transition="unclear",
                  order=None, critical_lambda=None, critical_occupation_jump=None,
                  scaling_exponent=None, pitchfork_like=None,
                  search_max_used=None, search_limit=None, error=None)
    try:
        if len(pattern) % 2:
            found = odd_critical_point(pattern)
            if found:
                critical, r_c, search_max = found
                is_second, jump, beta = odd_pitchfork_test(pattern, critical, r_c)
                result.update(critical_lambda=critical, search_max_used=search_max,
                              critical_occupation_jump=jump, scaling_exponent=beta,
                              pitchfork_like=is_second,
                              transition="second-order" if is_second else "unclear",
                              order=2 if is_second else None)
        else:
            found = even_critical_point(pattern)
            if found:
                critical, saddle, search_max = found
                jump = even_jump(pattern, critical, saddle)
                is_first = jump > JUMP_TOLERANCE
                result.update(critical_lambda=critical, search_max_used=search_max,
                              critical_occupation_jump=jump,
                              transition="first-order" if is_first else "continuous / unclear",
                              order=1 if is_first else None)

        if found is None:
            result.update(status="above-search-limit", transition="above-search-limit",
                          search_limit=MAX_LAMBDA)
    except Exception as exc:
        result.update(status="error", transition="error", error=repr(exc))
    return result


def random_patterns():
    """Generate distinct patterns of the requested primitive period."""
    if PERIOD_LENGTH < 1 or N_SAMPLES < 1 or DEGREE_MIN < 1 or DEGREE_MAX < DEGREE_MIN:
        raise ValueError("Invalid pattern sampling settings.")
    rng = np.random.default_rng(SEED)
    patterns = set()
    attempts = 0
    while len(patterns) < N_SAMPLES:
        attempts += 1
        if attempts > max(100_000, 100 * N_SAMPLES):
            raise RuntimeError("Could not generate enough distinct primitive patterns.")
        pattern = tuple(map(int, rng.integers(DEGREE_MIN, DEGREE_MAX + 1,
                                               size=PERIOD_LENGTH)))
        if len(minimal_period(pattern)) == PERIOD_LENGTH:
            patterns.add(pattern)
    return sorted(patterns)


FIELDS = ("input_pattern", "primitive_pattern", "period", "status", "transition",
          "order", "critical_lambda", "search_max_used", "search_limit",
          "critical_occupation_jump", "scaling_exponent", "pitchfork_like", "error")


def save_csv(filename, results):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results, elapsed):
    """Print counts, performance, and first-order occupation-jump statistics."""
    counts = {name: sum(r["transition"] == name for r in results)
              for name in ("first-order", "second-order", "unclear",
                           "continuous / unclear", "above-search-limit", "error")}
    total = len(results)
    throughput = total / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    for label, value in (
        ("Total", total),
        ("First-order", counts["first-order"]),
        ("Second-order", counts["second-order"]),
        ("Unclear", counts["unclear"] + counts["continuous / unclear"]),
        ("Above search limit", counts["above-search-limit"]),
        ("Errors", counts["error"]),
    ):
        print(f"{label:<21}: {value:,}")

    print()
    for label, value in (
        ("Runtime", f"{elapsed:.2f} sec"),
        ("Runtime", f"{elapsed / 60:.2f} min"),
        ("Runtime", f"{elapsed / 3600:.3f} hr"),
        ("Throughput", f"{throughput:.3f} patterns/sec"),
        ("Est. 100k", f"{100_000 / throughput / 60:.2f} min" if throughput else "N/A"),
        ("Est. 1M", f"{1_000_000 / throughput / 3600:.2f} hr" if throughput else "N/A"),
    ):
        print(f"{label:<21}: {value}")

    jumps = np.array([r["critical_occupation_jump"] for r in results
                      if r["transition"] == "first-order"
                      and r["critical_occupation_jump"] is not None
                      and np.isfinite(r["critical_occupation_jump"])], dtype=float)

    print("\n" + "=" * 75)
    print("CRITICAL OCCUPATION JUMP STATISTICS")
    print("=" * 75)
    if not len(jumps):
        print("No valid first-order jump values found.")
    else:
        for label, value in (
            ("Number of jumps", f"{len(jumps):,}"),
            ("Minimum jump", f"{jumps.min():.12g}"),
            ("Maximum jump", f"{jumps.max():.12g}"),
            ("Average jump", f"{jumps.mean():.12g}"),
            ("Median jump", f"{np.median(jumps):.12g}"),
        ):
            print(f"{label:<21}: {value}")


def main():
    patterns = random_patterns()
    print(f"Searching {len(patterns):,} primitive period-{PERIOD_LENGTH} patterns "
          f"with degrees {DEGREE_MIN}..{DEGREE_MAX} on {N_JOBS} workers")
    start = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(analyze_one)(pattern) for pattern in patterns)
    elapsed = time.perf_counter() - start

    print_summary(results, elapsed)

    interesting = [r for r in results if r["transition"] != "second-order"]
    for row in interesting[:100]:
        print(row["primitive_pattern"], row["transition"],
              "lambda=", row["critical_lambda"],
              "jump=", row["critical_occupation_jump"])

    save_csv(OUTPUT_FILE, results)
    save_csv(INTERESTING_FILE, interesting)
    print(f"Saved: {OUTPUT_FILE}, {INTERESTING_FILE}")


if __name__ == "__main__":
    main()
