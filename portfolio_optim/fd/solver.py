from __future__ import annotations

import math
import time

import numpy as np

from portfolio_optim.core.base import BaseSolver
from portfolio_optim.core.config import ConstraintConfig
from portfolio_optim.core.config import ExperimentConfig
from portfolio_optim.core.types import MarketParameters, SolverResult
from portfolio_optim.market.constraints import project_weights


def _normcdf_local(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def asymptotic_goal_value(
    wealth: np.ndarray,
    tau: float,
    sigma_eff: float,
    leverage_bound: float,
    target_wealth: float,
) -> np.ndarray:
    tau = max(tau, 1e-12)
    denom = max(leverage_bound * sigma_eff * math.sqrt(tau), 1e-12)
    z = np.minimum(0.0, np.log(np.maximum(wealth, 1e-12) / target_wealth)) / denom
    return 2.0 * _normcdf_local(z)


def sample_admissible_controls(
    n_assets: int,
    d: float,
    u: float,
    n_samples: int = 256,
    leverage_limit: float | None = None,
    long_only: bool | None = None,
    allow_cash: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample admissible multi-asset portfolios for the wealth-only HJB reduction.

    This is the multi-asset analogue of the scalar control interval [d, u]:
    we generate candidate portfolio weight vectors and project them back to the
    admissible set.
    """
    if long_only is None:
        long_only = d >= 0.0
    if leverage_limit is None:
        leverage_limit = max(1.0, n_assets * max(abs(d), abs(u)))

    constraints = ConstraintConfig(
        long_only=long_only,
        min_weight=d,
        max_weight=u,
        leverage_limit=leverage_limit,
        allow_cash=allow_cash,
    )
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.ones(n_assets), size=n_samples)
    if not long_only:
        base = base * rng.choice([-1.0, 1.0], size=base.shape)
    anchors = [np.zeros(n_assets)]
    anchors.extend(np.eye(n_assets))
    anchors.append(np.full(n_assets, 1.0 / n_assets))
    controls = np.vstack([np.asarray(anchors), base])
    projected = [project_weights(control, constraints) for control in controls]
    return np.unique(np.round(np.vstack(projected), decimals=12), axis=0)


def asymptotic_V_goalreach_multiasset(
    w_arr: np.ndarray,
    tau: float,
    covariance: np.ndarray,
    controls: np.ndarray,
    goal: float = 1.0,
) -> np.ndarray:
    """
    Multi-asset near-terminal asymptotic value.

    The scalar L*sigma term becomes the largest attainable instantaneous wealth
    volatility across admissible portfolios, estimated over `controls`.
    """
    sigma_eff = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", controls, covariance, controls), 1e-16))
    sigma_star = float(np.max(sigma_eff))
    return asymptotic_goal_value(w_arr, tau, sigma_star, 1.0, goal)


def fd_solve_viscosity_multiasset(
    mu: np.ndarray,
    r: float,
    covariance: np.ndarray,
    T: float,
    A: float,
    Nw: int,
    Nt: int,
    d: float,
    u: float,
    utility_fn,
    goal: float = 1.0,
    asymptotic_fn=None,
    UB: float = 0.0,
    UA: float | None = None,
    tau_asymp: float | None = None,
    record_taus: list[float] | None = None,
    n_control_samples: int = 256,
    leverage_limit: float | None = None,
    long_only: bool | None = None,
    allow_cash: bool = True,
    seed: int = 0,
    store_history: bool = False,
):
    """
    Multi-asset extension of the 1D viscosity-corrected wealth-grid solver.

    The state remains one-dimensional in wealth, while the control becomes a
    sampled set of admissible portfolio vectors `pi in R^n`. At each iteration
    we solve the same implicit tridiagonal system, but with drift/variance
    induced by the currently selected portfolio at each wealth node.
    """
    mu = np.asarray(mu, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n_assets = mu.size
    controls = sample_admissible_controls(
        n_assets=n_assets,
        d=d,
        u=u,
        n_samples=n_control_samples,
        leverage_limit=leverage_limit,
        long_only=long_only,
        allow_cash=allow_cash,
        seed=seed,
    )

    w = np.linspace(0.0, A, Nw + 1)
    dw = A / Nw
    dt = T / Nt
    wi = w[1:Nw]
    excess = mu - r

    if UA is None:
        UA = float(utility_fn(np.array([A])).flat[0])
    if tau_asymp is None:
        tau_asymp = 0.1 * T

    V = utility_fn(w).astype(float)
    snapshots: dict[float, np.ndarray] = {}
    policy_snaps: dict[float, np.ndarray] = {}
    value_history = np.zeros((Nt + 1, Nw + 1)) if store_history else None
    policy_history = np.zeros((Nt, Nw + 1, n_assets)) if store_history else None
    if store_history:
        value_history[-1] = V.copy()

    mu_eff = controls @ excess
    sigma_eff = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", controls, covariance, controls), 1e-16))

    def browne_V_local(w_arr: np.ndarray, tau_val: float) -> np.ndarray:
        tau_val = max(tau_val, 1e-10)
        safe_w = np.maximum(w_arr, 1e-10)
        safe_sig = np.maximum(sigma_eff, 1e-12)
        z = (
            np.log(safe_w[None, :] / goal) + (mu_eff[:, None] - 0.5 * safe_sig[:, None] ** 2) * tau_val
        ) / (safe_sig[:, None] * math.sqrt(tau_val))
        return _normcdf_local(z)

    if asymptotic_fn is None:
        asymptotic_fn = lambda wealth, tau: asymptotic_V_goalreach_multiasset(wealth, tau, covariance, controls, goal=goal)

    chosen_controls = np.zeros((Nw + 1, n_assets))
    for step in range(Nt):
        tau = (Nt - step) * dt
        V_old = V.copy()

        alpha = float(np.exp(-tau / tau_asymp))
        V_b = browne_V_local(w, tau)
        V_a = np.vstack([asymptotic_fn(w, tau) for _ in range(len(controls))])
        V_ws = alpha * V_a + (1.0 - alpha) * V_b
        best_idx = np.argmax(V_ws, axis=0)[1:Nw]
        chosen_controls[1:Nw] = controls[best_idx]

        for _it in range(60):
            prev_idx = best_idx.copy()
            mu_now = mu_eff[best_idx]
            sig_now = sigma_eff[best_idx]
            a2 = 0.5 * sig_now**2 * wi**2
            adv = mu_now * wi
            ap = np.maximum(adv, 0.0) / dw
            am = np.minimum(adv, 0.0) / dw
            a_s = -dt * (a2 / dw**2 - am)
            b_m = 1.0 + dt * (2.0 * a2 / dw**2 + ap - am)
            c_s = -dt * (a2 / dw**2 + ap)
            rhs = V_old[1:Nw].copy()
            rhs[0] -= a_s[0] * UB
            rhs[-1] -= c_s[-1] * UA
            a_s[0] = 0.0
            c_s[-1] = 0.0
            V_int = _thomas_solve(a_s, b_m, c_s, rhs)
            V_new = np.empty(Nw + 1)
            V_new[0] = UB
            V_new[Nw] = UA
            V_new[1:Nw] = V_int

            Vww = (V_new[2:] - 2.0 * V_new[1:-1] + V_new[:-2]) / dw**2
            Vw = (V_new[2:] - V_new[:-2]) / (2.0 * dw)
            H = (
                0.5 * sigma_eff[:, None] ** 2 * wi[None, :] ** 2 * Vww[None, :]
                + mu_eff[:, None] * wi[None, :] * Vw[None, :]
            )
            best_idx = np.argmax(H, axis=0)
            chosen_controls[1:Nw] = controls[best_idx]
            V = V_new
            if np.array_equal(best_idx, prev_idx):
                break

        chosen_controls[0] = controls[0]
        chosen_controls[-1] = controls[best_idx[-1]] if best_idx.size else controls[0]
        if store_history:
            value_history[-(step + 2)] = V.copy()
            policy_history[-(step + 1)] = chosen_controls.copy()

        if record_taus:
            tau_prev = (Nt - step + 1) * dt
            for tr in record_taus:
                if tau_prev >= tr >= tau and tr not in snapshots:
                    snapshots[tr] = V.copy()
                    policy_snaps[tr] = chosen_controls.copy()

    if record_taus:
        snapshots[0.0] = V.copy()
        policy_snaps[0.0] = chosen_controls.copy()

    return (
        w,
        V,
        chosen_controls,
        snapshots,
        policy_snaps,
        controls,
        value_history,
        policy_history,
    )


def _thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(b)
    c2 = np.zeros(n)
    d2 = np.zeros(n)
    x = np.zeros(n)
    c2[0] = c[0] / b[0]
    d2[0] = rhs[0] / b[0]
    for k in range(1, n):
        denom = b[k] - a[k] * c2[k - 1]
        c2[k] = c[k] / denom if k < n - 1 else 0.0
        d2[k] = (rhs[k] - a[k] * d2[k - 1]) / denom
    x[-1] = d2[-1]
    for k in range(n - 2, -1, -1):
        x[k] = d2[k] - c2[k] * x[k + 1]
    return x


class FiniteDifferencePortfolioSolver(BaseSolver):
    """
    Viscosity-corrected backward Euler wealth-grid solver.

    For `n_assets=1`, this follows the notebook structure closely:
    - asymptotic Eq. (18) warmstart
    - policy iteration
    - tridiagonal implicit solve

    For `n_assets>1`, we keep the same corrected wealth-grid scheme but replace the
    scalar control with sampled multi-asset portfolios. This preserves the benchmark's
    terminal asymptotic correction while remaining computationally usable.
    """

    method_family = "fd"
    method_name = "fd_hjb_viscosity"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._wealth_grid: np.ndarray | None = None
        self._value_grid: np.ndarray | None = None
        self._policy_grid: np.ndarray | None = None
        self._controls: np.ndarray | None = None

    def _policy_from_value_1d(
        self,
        V: np.ndarray,
        wealth_grid: np.ndarray,
        eta: float,
        sigma: float,
    ) -> np.ndarray:
        dw = wealth_grid[1] - wealth_grid[0]
        wi = wealth_grid[1:-1]
        Vww = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / (dw**2)
        Vw = (V[2:] - V[:-2]) / (2.0 * dw)
        d = self.config.problem.constraints.min_weight
        u = self.config.problem.constraints.max_weight
        sig2 = sigma**2
        f_d = 0.5 * d**2 * wi**2 * sig2 * Vww + d * wi * eta * Vw
        f_u = 0.5 * u**2 * wi**2 * sig2 * Vww + u * wi * eta * Vw
        safe = np.where(Vww < -1e-12, Vww, -1e-12)
        pi_int = np.clip(-(wi * eta * Vw) / (wi**2 * sig2 * safe + 1e-12), d, u)
        pi = np.where(Vww < -1e-12, pi_int, np.where(f_u >= f_d, u, d))
        full = np.zeros((wealth_grid.size, 1))
        full[1:-1, 0] = pi
        full[0, 0] = d
        full[-1, 0] = u if eta >= 0 else d
        return full

    def _fit_single_asset_exact(self, market: MarketParameters, seed: int) -> SolverResult:
        del seed
        start = time.perf_counter()
        wealth_grid = np.linspace(self.config.fd.wealth_min, self.config.fd.wealth_max, self.config.fd.n_wealth_points + 1)
        dt = 1.0
        dw = wealth_grid[1] - wealth_grid[0]
        wi = wealth_grid[1:-1]
        target = self.config.problem.target_wealth
        V = self._terminal_utility(wealth_grid)
        value_hist = np.zeros((self.config.problem.horizon_steps + 1, wealth_grid.size))
        policy_hist = np.zeros((self.config.problem.horizon_steps, wealth_grid.size, 1))
        value_hist[-1] = V.copy()

        mu = float(market.mean_returns[0] + self.config.problem.risk_free_rate / market.periods_per_year)
        r = float(self.config.problem.risk_free_rate / market.periods_per_year)
        sigma = float(np.sqrt(market.covariance[0, 0]))
        eta = mu - r
        d = self.config.problem.constraints.min_weight
        u = self.config.problem.constraints.max_weight
        tau_asymp = max(1.0, 0.08 * self.config.problem.horizon_steps)

        def browne_local(w_arr: np.ndarray, tau_val: float) -> np.ndarray:
            tau_val = max(tau_val, 1e-10)
            z = (
                np.log(np.maximum(w_arr, 1e-12) / target)
                + (eta - 0.5 * sigma**2) * tau_val
            ) / (sigma * math.sqrt(tau_val))
            return _normcdf_local(z)

        for step in range(self.config.problem.horizon_steps):
            tau = (self.config.problem.horizon_steps - step) * dt
            V_old = V.copy()
            alpha = float(np.exp(-tau / tau_asymp))
            V_b = browne_local(wealth_grid, tau)
            V_a = asymptotic_goal_value(wealth_grid, tau, sigma, max(u, -d), target)
            V_ws = alpha * V_a + (1.0 - alpha) * V_b
            pi_grid = self._policy_from_value_1d(V_ws, wealth_grid, eta, sigma)

            for _ in range(60):
                old_pi = pi_grid.copy()
                pi = pi_grid[1:-1, 0]
                a2 = 0.5 * pi**2 * wi**2 * sigma**2
                adv = pi * wi * eta
                ap = np.maximum(adv, 0.0) / dw
                am = np.minimum(adv, 0.0) / dw
                a_sub = -dt * (a2 / dw**2 - am)
                b_mid = 1.0 + dt * (2.0 * a2 / dw**2 + ap - am)
                c_sup = -dt * (a2 / dw**2 + ap)
                rhs = V_old[1:-1].copy()
                rhs[0] -= a_sub[0] * 0.0
                rhs[-1] -= c_sup[-1] * 1.0
                a_sub[0] = 0.0
                c_sup[-1] = 0.0
                V_inner = _thomas_solve(a_sub, b_mid, c_sup, rhs)
                V_new = np.empty_like(V)
                V_new[0] = 0.0
                V_new[-1] = 1.0
                V_new[1:-1] = V_inner
                pi_grid = self._policy_from_value_1d(V_new, wealth_grid, eta, sigma)
                if np.max(np.abs(pi_grid - old_pi)) < 1e-8:
                    V = V_new
                    break
                V = V_new

            value_hist[-(step + 2)] = V
            policy_hist[-(step + 1)] = pi_grid

        solve_time = time.perf_counter() - start
        self._wealth_grid = wealth_grid
        self._value_grid = value_hist
        self._policy_grid = policy_hist
        return SolverResult(
            method_family=self.method_family,
            method_name=self.method_name,
            n_assets=1,
            seed=0,
            solve_time_sec=solve_time,
            wealth_grid=wealth_grid,
            value_grid=value_hist,
            policy_grid=policy_hist,
            metadata={
                "tickers": market.tickers,
                "scheme": "exact notebook-style asymptotic viscosity correction",
                "dimension_mode": "single_asset_exact",
            },
        )

    def _terminal_utility(self, wealth: np.ndarray) -> np.ndarray:
        return (np.asarray(wealth, dtype=float) >= self.config.problem.target_wealth).astype(float)

    def fit(self, market: MarketParameters, n_assets: int, seed: int) -> SolverResult:
        if n_assets == 1:
            result = self._fit_single_asset_exact(market, seed)
            result.seed = seed
            return result

        start = time.perf_counter()
        r = float(self.config.problem.risk_free_rate / market.periods_per_year)
        wealth_grid, _, _, _, _, controls, value_hist, policy_hist = fd_solve_viscosity_multiasset(
            mu=market.mean_returns,
            r=r,
            covariance=market.covariance,
            T=float(self.config.problem.horizon_steps),
            A=self.config.fd.wealth_max,
            Nw=self.config.fd.n_wealth_points,
            Nt=self.config.problem.horizon_steps,
            d=self.config.problem.constraints.min_weight,
            u=self.config.problem.constraints.max_weight,
            utility_fn=self._terminal_utility,
            goal=self.config.problem.target_wealth,
            UB=0.0,
            UA=1.0,
            tau_asymp=max(1.0, 0.08 * self.config.problem.horizon_steps),
            n_control_samples=self.config.fd.n_control_samples,
            leverage_limit=self.config.problem.constraints.leverage_limit,
            long_only=self.config.problem.constraints.long_only,
            allow_cash=self.config.problem.constraints.allow_cash,
            seed=seed,
            store_history=True,
        )

        solve_time = time.perf_counter() - start
        self._wealth_grid = wealth_grid
        self._value_grid = value_hist
        self._policy_grid = policy_hist
        self._controls = controls
        return SolverResult(
            method_family=self.method_family,
            method_name=self.method_name,
            n_assets=n_assets,
            seed=seed,
            solve_time_sec=solve_time,
            wealth_grid=wealth_grid,
            value_grid=value_hist,
            policy_grid=policy_hist,
            metadata={
                "tickers": market.tickers,
                "scheme": "viscosity-corrected multi-asset extension",
                "control_count": int(len(controls)),
                "dimension_mode": "multi_asset_extension",
            },
        )

    def policy(self, t_index: int, wealth: float, market_features: np.ndarray | None = None) -> np.ndarray:
        if self._wealth_grid is None or self._policy_grid is None:
            raise RuntimeError("Call fit before policy.")
        t_index = min(max(t_index, 0), self._policy_grid.shape[0] - 1)
        idx = int(np.clip(np.searchsorted(self._wealth_grid, wealth), 0, len(self._wealth_grid) - 1))
        return self._policy_grid[t_index, idx]
