# GMG implementation settings

The GPU V-cycle allocates residual, restricted RHS, and coarse correction buffers
once per level in `init_for_pcg()`. Reinitializing rebuilds these buffers and the
GPU matrices. Residual and restriction overwrite their buffers; coarse corrections
are cleared on every cycle. A solver instance and its workspaces must be used
sequentially, not by concurrent solves.

## Diagnostics

`VEM_GMGSolver::diagnostic_checks` enables intermediate vector norm checks in the
V-cycle. It defaults to `false` because these checks add GPU reductions and host
synchronization.

The active GMG path uses AMGCL's restarted right-preconditioned GMRES. It has the
same outer method, tolerance (`1e-3`), maximum iteration count (`10000`), and
restart length (`30`) as the AMGCL comparison path. GMRES provides its own residual
test; its iteration logging is disabled in the demo.

The active smoke loop prints `[Timer] GMG_GMRES_build` once and `[Timer]
GMG_GMRES_solve` once per pressure solve. The solve timer includes GMRES, every
GMG V-cycle, and the final device-to-host pressure copy.

## Coarse solve

Configure these fields before `init_for_pcg()`:

```cpp
solver.direct_coarse = false;
solver.coarse_smoothing_iterations = 8;
solver.pre_smoothing_iterations = 1;
solver.post_smoothing_iterations = 1;
```

The GPU path snapshots both settings during initialization, so the preconditioner
does not change between GMRES iterations. The active demo uses 8 coarse Chebyshev
applications and one pre-/post-smoothing application on every other level. Fixed
counts such as 8, 16, and 30 can be compared later using total solve time and true
residual. No faster count has been established.

With `direct_coarse = true`, the backend direct solver is constructed once and
reused. The unused AMGCL coarse hierarchy is no longer constructed. Direct mode
still requires a suitable coarse matrix, including proper treatment of any
pressure nullspace. Changing mode or count requires reinitialization. The legacy
Eigen-vector V-cycle uses `coarse_smoothing_iterations` for its smoothing path.

## Chebyshev buffers

The bundled Chebyshev implementation already stores its recurrence vectors and
diagonal scaling inside the relaxation object. GPU smoothing reuses that object
without per-call vector allocation. The Eigen-vector wrapper now also reuses its
host staging arrays and lazily allocated device transfer buffers. These are reset
when the smoother is rebuilt.

## Device initial guess

The default `VEX_GMRES::solve()` behavior still uploads the supplied host initial
guess. The active GMG smoke demo explicitly uses `InitialGuess::Device`: the first
solve uploads the host guess, and subsequent successful solves retain pressure
on the GPU. Results are still downloaded for CPU pressure application.

Call `reset_device_initial_guess()` after changing pressure on the CPU or switching
simulation state. `build()` and `clear()` invalidate the cached guess automatically.
After changing the matrix or hierarchy, rebuild the GMG and GMRES objects together.
The solve API's optional fourth argument preserves existing three-argument callers.

These changes have only been statically reviewed; they have not been compiled,
run, or benchmarked.
