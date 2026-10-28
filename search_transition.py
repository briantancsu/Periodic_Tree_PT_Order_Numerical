"""Search periodic hard-core tree recursions for bifurcations without parity-based labels.

For a primitive degree pattern p=(d1,...,dk), put
    F_p(r) = f_d1 o f_d2 o ... o f_dk (r),
    f_d(r) = lambda/(1+r)^d.
The ONLY parity-based rule is construction of the increasing map g:
    k even: g = F_p,
    k odd:  g = F_p o F_p.

All patterns use the SAME critical point equations:
    g_lambda(r*)=r*,  dg/dr(lambda,r*)=1.
Classify candidate bifurcations by their derivatives:
    saddle-node: d2g/dr2 != 0, dg/dlambda != 0
    pitchfork:   d2g/dr2  = 0, d3g/dr3 != 0,
                 dg/dlambda = 0, d2g/(dr dlambda) != 0.

All searches use one fixed logarithmic scan from LAMBDA_MIN to LAMBDA_MAX;
there is no adaptive expansion of the lambda range.

The numerical classes are *bifurcation candidates*, not rigorous proofs of
thermodynamic first/second-order phase transitions. An unsuccessful scan or a
degenerate/poorly resolved point stays inconclusive.

Requires: numpy, scipy, joblib. Example: python search_transition_unbiased.py
"""

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import csv
import time
import numpy as np
from scipy.optimize import brentq, least_squares
from joblib import Parallel, delayed, parallel_config

# ------------------------ User settings ------------------------
PERIOD_LENGTH = 10           # Change to 10, etc.
N_SAMPLES = 15_000
DEGREE_MIN, DEGREE_MAX = 2, 100
SEED = 1234
N_JOBS = -1

LAMBDA_MIN = 1e-10
LAMBDA_MAX = 1000.0      # Fixed upper bound; change this as needed.
SCAN_POINTS = 160         # Log-spaced lambda values across the full interval.
R_GRID_SIZE = 700

# Dimensionless derivative tolerances; use same thresholds for all periods.
CURVATURE_TOL = 2e-4
CUBIC_TOL = 1e-5
PARAMETER_TOL = 2e-4
MIXED_TOL = 1e-5
SOLVER_TOL = 1e-10

OUTPUT_FILE = "transition_results.csv"
INTERESTING_FILE = "interesting_patterns.csv"


# ---------------------- Composition ---------------------------
def minimal_period(pattern):
    pattern = tuple(map(int, pattern))
    if not pattern or min(pattern) <= 0:
        raise ValueError("Degree patterns must contain positive integers")
    for length in range(1, len(pattern) + 1):
        if len(pattern) % length == 0 and pattern[:length] * (len(pattern)//length) == pattern:
            return pattern[:length]
    raise RuntimeError("Could not find primitive period")


def increasing_pattern(primitive):
    """The only parity-dependent choice: square an odd-length composition."""
    return primitive if len(primitive) % 2 == 0 else primitive + primitive


def g_eval(degrees, lam, r, derivatives=0):
    """Compute g, g', g'', g''' analytically by repeated chain rule.

    `degrees` is already the increasing composition (not the primitive one).
    Works on scalars and NumPy arrays; derivatives=0/1/2/3.
    """
    y = np.asarray(r, dtype=float)
    if derivatives >= 1:
        p = np.ones_like(y)
    if derivatives >= 2:
        q = np.zeros_like(y)
    if derivatives >= 3:
        t = np.zeros_like(y)

    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        for degree in reversed(degrees):
            inv = 1.0 / (1.0 + y)
            next_y = np.exp(np.log(lam) - degree * np.log1p(y))
            if derivatives >= 1:
                f1 = -degree * next_y * inv
            if derivatives >= 2:
                f2 = degree * (degree + 1) * next_y * inv**2
            if derivatives >= 3:
                f3 = -degree * (degree + 1) * (degree + 2) * next_y * inv**3
                next_t = f3*p**3 + 3.0*f2*p*q + f1*t
            if derivatives >= 2:
                next_q = f2*p**2 + f1*q
            if derivatives >= 1:
                next_p = f1*p
            y = next_y
            if derivatives >= 1:
                p = next_p
            if derivatives >= 2:
                q = next_q
            if derivatives >= 3:
                t = next_t

    if derivatives == 0:
        return y
    if derivatives == 1:
        return y, p
    if derivatives == 2:
        return y, p, q
    return y, p, q, t


def occupation(r):
    return r/(1.0+r)


# ---------------- Find fixed point branches --------------------
def fixed_points(degrees, lam):
    """Sign-changing roots of g(r)-r in [0,lambda]. Tangencies are refined later.

    Use a linear + logarithmic grid in r/lambda, to resolve small ratios.
    """
    linear = np.linspace(0.0, 1.0, R_GRID_SIZE)
    log_part = np.geomspace(1e-100, 1.0, max(400, R_GRID_SIZE))
    x_grid = np.unique(np.r_[linear, log_part])
    h = g_eval(degrees, lam, lam*x_grid)/lam - x_grid
    finite = np.isfinite(h)
    crossing = np.flatnonzero(finite[:-1] & finite[1:] & (h[:-1]*h[1:] < 0))

    def h_scalar(x):
        return float(g_eval(degrees, lam, lam*x)/lam - x)

    roots = []
    for i in crossing:
        root_x = brentq(h_scalar, x_grid[i], x_grid[i+1], xtol=1e-300, rtol=1e-12)
        root = lam*root_x
        if not roots or abs(root-roots[-1]) > 1e-8*max(abs(root),abs(roots[-1]),1e-100):
            roots.append(root)
    return roots


def find_three_roots(degrees):
    """Scan only [LAMBDA_MIN, LAMBDA_MAX] for three fixed points of g.

    The same fixed logarithmic lambda grid is used for every pattern.
    """
    for lam in np.geomspace(LAMBDA_MIN, LAMBDA_MAX, SCAN_POINTS):
        roots = fixed_points(degrees, float(lam))
        if len(roots) >= 3:
            return float(lam), roots
    return None


def refine_critical(degrees, lam_hit, roots):
    """Find tangency g(r)=r, g'(r)=1, seeding from neighboring roots."""
    candidates = []
    for j in range(len(roots)-1):
        # Close roots signal a nearby saddle-node; also works for pitchfork.
        candidates.append(np.sqrt(roots[j]*roots[j+1]))
    candidates += list(roots[1:-1])
    candidates = list(dict.fromkeys(float(r) for r in candidates if 0 < r < lam_hit))

    def equations(z):
        lam, r = np.exp(z)
        value, slope = g_eval(degrees, lam, r, derivatives=1)
        return [float((value-r)/max(r, 1e-300)), float(slope-1.0)]

    best = None
    for r_init in candidates:
        sol = least_squares(
            equations, x0=np.log([lam_hit, r_init]),
            bounds=(np.log([LAMBDA_MIN, 1e-300]), np.log([LAMBDA_MAX, LAMBDA_MAX])),
            max_nfev=300, xtol=1e-13, ftol=1e-13, gtol=1e-13,
        )
        lam, r = np.exp(sol.x)
        resid = float(np.linalg.norm(sol.fun))
        if np.isfinite(resid) and 0 < r < lam and LAMBDA_MIN <= lam <= min(LAMBDA_MAX, lam_hit*(1+1e-5)):
            if best is None or resid < best[2]:
                best = (float(lam), float(r), resid)
            if resid < 1e-10:
                break
    if best is None:
        return None

    # A pitchfork is degenerate for the two-equation tangency solver:
    # g-r and g'-1 can be tiny slightly away from the true critical point.
    # Try the additional equation r*g''=0 *only* when the numerically found
    # point is near-degenerate. The same check is applied to every pattern,
    # without inspecting period parity. Adopt this refinement only if ALL
    # three residuals become tiny; do not force saddle-nodes into pitchforks.
    lam0, r0, _ = best
    v, slope, curvature = g_eval(degrees, lam0, r0, derivatives=2)
    if np.isfinite(curvature) and (abs(r0*curvature) < 0.05 or best[2] > 1e-11):
        def triple_equations(z):
            lam, r = np.exp(z)
            v, p, q = g_eval(degrees, lam, r, derivatives=2)
            return [(float(v)-r)/max(r, 1e-300), float(p)-1.0, float(r*q)]

        refined = least_squares(triple_equations, np.log([lam0, r0]),
                                max_nfev=500, xtol=1e-13, ftol=1e-13, gtol=1e-13)
        lam, r = np.exp(refined.x)
        resid = float(np.linalg.norm(refined.fun))
        if (np.isfinite(resid) and resid < 1e-9
                and LAMBDA_MIN <= lam <= min(LAMBDA_MAX, lam_hit*(1+1e-5)) and 0 < r < lam):
            return float(lam), float(r), resid

    return best if best[2] <= SOLVER_TOL else None


# ------------------ Uniform classification --------------------
def classify_bifurcation(degrees, critical, r_c):
    """Apply exactly the same second/third-derivative tests to every pattern."""
    g, first, second, third = g_eval(degrees, critical, r_c, derivatives=3)
    # Scale by r_c so units are comparable across very different lambda values.
    curvature = float(r_c*second)
    cubic = float(r_c*r_c*third)

    # Parameter derivatives are taken at fixed r, not along the fixed-point branch.
    eps = 2e-5
    gp, dp = g_eval(degrees, critical*(1+eps), r_c, derivatives=1)
    gm, dm = g_eval(degrees, critical*(1-eps), r_c, derivatives=1)
    partial_lambda = float((gp-gm)/(2*eps*max(r_c, 1e-300)))  # lambda*g_lambda/r_c
    mixed = float((dp-dm)/(2*eps))                             # lambda*g_{r lambda}

    details = dict(second_derivative=float(second), third_derivative=float(third),
                   scaled_curvature=curvature, scaled_cubic=cubic,
                   scaled_parameter_derivative=partial_lambda,
                   scaled_mixed_derivative=mixed)
    if not all(np.isfinite(v) for v in details.values()):
        return "unclear", details

    # For saddle-node candidates, the independent derivative is nonzero.
    if abs(curvature) > CURVATURE_TOL and abs(partial_lambda) > PARAMETER_TOL:
        return "first-order", details

    # For pitchfork candidates, g''=0 is not sufficient by itself.
    if (abs(curvature) <= CURVATURE_TOL and abs(cubic) > CUBIC_TOL
            and abs(partial_lambda) <= PARAMETER_TOL
            and abs(mixed) > MIXED_TOL):
        return "second-order", details

    return "unclear", details


def candidate_jump(degrees, critical, r_c, kind):
    if kind == "second-order":
        return 0.0  # pitchfork branches meet continuously at criticality
    if kind != "first-order":
        return None
    # Estimate the jump using the UNIQUE surviving fixed point just BELOW
    # the bifurcation. Solving exactly at a saddle-node is fragile because a
    # double root can look like two sign-changing roots under roundoff.
    lam_below = critical*(1.0-1e-5)
    if lam_below <= 0:
        return None
    try:
        x = brentq(lambda z: float(g_eval(degrees, lam_below, lam_below*z)/lam_below-z),
                   0.0, 1.0, xtol=1e-300, rtol=1e-12)
        return float(abs(occupation(lam_below*x)-occupation(r_c)))
    except (ValueError, OverflowError):
        return None


def analyze_one(pattern):
    original = tuple(map(int, pattern))
    primitive = minimal_period(original)
    degrees = increasing_pattern(primitive)
    row = dict(input_pattern=original, primitive_pattern=primitive,
               primitive_length=len(primitive), composition_length=len(degrees),
               status="ok", transition="unclear", order=None, critical_lambda=None,
               critical_ratio=None, critical_occupation_jump=None,
               second_derivative=None, third_derivative=None,
               scaled_curvature=None, scaled_cubic=None,
               scaled_parameter_derivative=None, scaled_mixed_derivative=None,
               solver_residual=None, detected_at_lambda=None,
               search_limit=None, error=None)
    try:
        scan = find_three_roots(degrees)
        if scan is None:
            row.update(status="above-search-limit", transition="above-search-limit",
                       search_limit=LAMBDA_MAX)
            return row
        lam_hit, roots = scan
        row.update(detected_at_lambda=lam_hit)
        critical = refine_critical(degrees, lam_hit, roots)
        if critical is None:
            row.update(status="unresolved-critical", transition="unclear")
            return row
        lam_c, r_c, residual = critical
        kind, details = classify_bifurcation(degrees, lam_c, r_c)
        row.update(critical_lambda=lam_c, critical_ratio=r_c,
                   solver_residual=residual, transition=kind,
                   order={"first-order":1, "second-order":2}.get(kind),
                   critical_occupation_jump=candidate_jump(degrees,lam_c,r_c,kind),
                   **details)
    except Exception as exc:
        row.update(status="error", transition="error", error=repr(exc))
    return row


# --------------------- Sample and outputs ---------------------
def random_patterns():
    if PERIOD_LENGTH < 1 or N_SAMPLES < 1 or DEGREE_MIN < 1 or DEGREE_MAX < DEGREE_MIN:
        raise ValueError("Invalid search settings")
    if not (0 < LAMBDA_MIN < LAMBDA_MAX) or SCAN_POINTS < 2:
        raise ValueError("Need 0 < LAMBDA_MIN < LAMBDA_MAX and SCAN_POINTS >= 2")
    rng = np.random.default_rng(SEED)
    patterns = set()
    attempts = 0
    while len(patterns) < N_SAMPLES:
        attempts += 1
        if attempts > max(100_000, 100*N_SAMPLES):
            raise RuntimeError("Could not generate enough distinct primitive patterns")
        p = tuple(map(int, rng.integers(DEGREE_MIN, DEGREE_MAX+1, size=PERIOD_LENGTH)))
        if len(minimal_period(p)) == PERIOD_LENGTH:
            patterns.add(p)
    return sorted(patterns)


FIELDS = ("input_pattern", "primitive_pattern", "primitive_length", "composition_length",
          "status", "transition", "order", "critical_lambda", "critical_ratio",
          "critical_occupation_jump", "second_derivative", "third_derivative",
          "scaled_curvature", "scaled_cubic", "scaled_parameter_derivative",
          "scaled_mixed_derivative", "solver_residual", "detected_at_lambda",
          "search_limit", "error")


def save_csv(filename, results):
    with open(filename, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results, elapsed):
    from collections import Counter
    counts = Counter(row["transition"] for row in results)
    total = len(results)
    throughput = total/elapsed if elapsed > 0 else 0
    print("\n" + "="*75 + "\nSUMMARY\n" + "="*75)
    for label, value in (("Total", total), ("First-order", counts["first-order"]),
                         ("Second-order", counts["second-order"]),
                         ("Unclear", counts["unclear"]),
                         ("Above search limit", counts["above-search-limit"]),
                         ("Errors", counts["error"])):
        print(f"{label:<21}: {value:,}")
    print("\n" + "="*75 + "\nPERFORMANCE\n" + "="*75)
    for label, value in (("Runtime (sec)", f"{elapsed:.2f}"),
                         ("Runtime (min)", f"{elapsed/60:.2f}"),
                         ("Throughput", f"{throughput:.2f} patterns/sec"),
                         ("Est. 100k", f"{100_000/throughput/60:.2f} min" if throughput else "N/A"),
                         ("Est. 1M", f"{1_000_000/throughput/3600:.2f} hr" if throughput else "N/A")):
        print(f"{label:<21}: {value}")
    jumps = np.array([row["critical_occupation_jump"] for row in results
                      if row["transition"] == "first-order"
                      and row["critical_occupation_jump"] is not None], dtype=float)
    print("\n" + "="*75 + "\nCRITICAL OCCUPATION JUMP STATISTICS\n" + "="*75)
    if jumps.size:
        for label, val in (("Number of jumps", jumps.size), ("Minimum jump", jumps.min()),
                           ("Maximum jump", jumps.max()), ("Average jump", jumps.mean()),
                           ("Median jump", np.median(jumps))):
            print(f"{label:<21}: {val:.12g}" if isinstance(val, float) or isinstance(val,np.floating)
                  else f"{label:<21}: {val:,}")
    else:
        print("No resolved first-order jumps.")


def main():
    patterns = random_patterns()
    print(f"Searching {len(patterns):,} primitive period-{PERIOD_LENGTH} patterns "
          f"(degrees {DEGREE_MIN}..{DEGREE_MAX}) using {N_JOBS} workers")
    print(f"Fixed lambda interval: [{LAMBDA_MIN:g}, {LAMBDA_MAX:g}], "
          f"{SCAN_POINTS} log-spaced samples")
    start = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(analyze_one)(p) for p in patterns)
    elapsed = time.perf_counter()-start
    print_summary(results, elapsed)
    save_csv(OUTPUT_FILE, results)
    save_csv(INTERESTING_FILE, [r for r in results if r["transition"] != "first-order"])
    print(f"\nSaved: {OUTPUT_FILE}, {INTERESTING_FILE}")


if __name__ == '__main__':
    main()
