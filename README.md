# Snowball Structured Product Pricing & Risk Analytics with Monte Carlo and Hull–White

## Overview

This project develops a numerical framework for pricing and analysing **path-dependent Snowball / ABRC structured products**.

The model combines:

- Monte Carlo simulation of the underlying asset
- Knock-in / knock-out barrier logic
- Monthly observation schedules
- Stochastic interest-rate simulation using a Hull–White mean-reverting process
- Theoretical coupon calibration
- Finite-difference Greeks
- Path simulation and barrier visualisation
- Pathwise payoff distributions
- Value-at-Risk and Expected Shortfall
- Interest-rate stress testing

The overall workflow is:

```text
Structured Product Terms
        |
        v
Trading Calendar & Monitoring Schedule
        |
        v
Underlying Price Simulation
        +
Hull–White Rate Simulation
        |
        v
Path-Dependent KI / KO Payoff Logic
        |
        v
Monte Carlo Valuation
        |
        v
Theoretical Coupon Calibration
        |
        v
Delta / Gamma / Vega / Theta / Rho
        |
        v
Pathwise Payoff Distribution
        |
        v
VaR / Expected Shortfall
        |
        v
Interest-Rate Stress Testing
```

---

# Product Structure

The Snowball contract is represented through a set of path-dependent conditions.

At inception, the contract specifies:

- Initial underlying price
- Knock-in barrier
- Knock-out barrier
- Knock-out observation dates
- Coupon rate
- Maturity date
- Volatility assumption
- Dividend yield
- Risk-free rate
- Notional and margin ratio

The payoff depends not only on the terminal underlying price, but also on whether barrier events occurred during the contract life.

---

# Example: CSI 500 Snowball

The baseline parameter set uses a CSI 500-linked ABRC structure.

| Parameter | Value |
|---|---:|
| Initial Underlying Price | 5,400 |
| Knock-Out Ratio | 103% |
| Knock-In Ratio | 75% |
| Knock-Out Observations | 21 |
| Assumed Coupon | 20% |
| Annualised Volatility | 14% |
| Continuous Dividend Yield | 6.3% |
| Risk-Free Rate | 1.5% |
| Contract Start | 2 Jan 2024 |
| Final Observation | 31 Dec 2025 |
| Hull–White Mean Reversion | 0.10 |
| Hull–White Rate Volatility | 0.01 |

The knock-out level is:

$$
B_{KO} = 1.03 \times 5400
$$

and the knock-in level is:

$$
B_{KI} = 0.75 \times 5400
$$

The contract contains 21 scheduled knock-out observation dates over the life of the product.

---

# Trading Calendar

The model constructs a trading calendar using the Shanghai Stock Exchange calendar.

Calendar handling is required because barrier observations and stochastic price evolution are based on actual trading days rather than simple calendar-day increments.

The simulation therefore separates:

```text
Calendar Days
      |
      v
Exchange Trading Days
      |
      v
Monitoring Dates
      |
      v
Monte Carlo Observation Grid
```

The knock-out monitoring dates are mapped directly onto the generated trading calendar.

---

# Underlying Price Simulation

The underlying asset follows a geometric Brownian motion process.

For each trading step:

$$
S_{t+\Delta t}
=
S_t
\exp
\left[
\left(
r-q-\frac{1}{2}\sigma^2
\right)\Delta t
+
\sigma\sqrt{\Delta t}Z_t
\right]
$$

where:

- $S_t$ = current underlying price
- $r$ = risk-free rate
- $q$ = continuous dividend yield
- $\sigma$ = underlying volatility
- $Z_t$ = standard-normal shock

The default Monte Carlo engine uses:

```text
300,000 simulated paths
```

for product valuation.

PyTorch tensors are used to accelerate the numerical simulation and payoff calculation.

The computational device is selected automatically:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

# Hull–White Interest-Rate Simulation

The framework additionally simulates stochastic short-rate paths using a mean-reverting Hull–White-style process.

The implementation follows:

$$
r_{t+\Delta t}
=
r_t
+
a(r_0-r_t)\Delta t
+
\sigma_r\sqrt{\Delta t}Z_t
$$

where:

- $r_t$ = current short rate
- $r_0$ = initial / long-run reference rate
- $a$ = mean-reversion speed
- $\sigma_r$ = short-rate volatility

The baseline parameters are:

```text
Mean reversion:       a = 0.10
Rate volatility:      sigma_r = 0.01
Initial rate:         r0 = 1.5%
```

The simulated rate path is then used to construct path-dependent discount factors.

Conceptually:

$$
DF_t
=
\exp
\left(
-\sum_{\tau=1}^{t}r_{\tau}\Delta t
\right)
$$

This allows product cash flows to be discounted under stochastic interest-rate scenarios rather than using only a single deterministic discount curve.

---

# Path-Dependent Payoff Logic

Each Monte Carlo trajectory is classified according to its barrier history.

## 1. Knock-Out Path

At each scheduled monitoring date, the model checks whether:

$$
S_t \geq B_{KO,t}
$$

If a knock-out occurs, the first qualifying observation date determines the early redemption point.

The contract then receives the corresponding accrued knock-out coupon and exits early.

---

## 2. No Knock-Out and No Knock-In

If the product survives all knock-out observation dates and the underlying never breaches the knock-in barrier, the contract reaches maturity and receives the maturity coupon.

---

## 3. Knock-In Without Knock-Out

If the underlying breaches the knock-in barrier but never subsequently satisfies a knock-out condition, the maturity payoff can become linked to the terminal performance of the underlying.

The downside component is represented by:

$$
\frac{S_T}{S_0}-1
$$

when the relevant knock-in and maturity conditions are satisfied.

---

# Monte Carlo Valuation

After applying the payoff logic to all simulated paths, the product value is estimated as the average discounted path payoff:

$$
V
=
\frac{1}{M}
\sum_{i=1}^{M}
Payoff_i
$$

where $M$ is the number of simulated Monte Carlo paths.

For the baseline CSI 500 parameter set with a 20% assumed coupon, the model produces an expected payoff of approximately:

```text
0.03740
```

---

# Theoretical Coupon Calibration

The project also solves the inverse pricing problem:

> What coupon makes the simulated contract value equal to a target value?

A binary-search routine repeatedly reprices the product under different coupon assumptions.

The calibration solves approximately:

$$
V(c) = V_{target}
$$

where $c$ is the coupon rate.

The search continues until the coupon interval is smaller than:

```text
1e-5
```

---

## CSI 500 Result

Using the CSI 500 parameter set:

```text
Theoretical Coupon Rate = 12.7064%
```

---

## BTC Snowball Example

The same framework is also applied to a higher-volatility BTC-linked Snowball.

The BTC example uses:

| Parameter | Value |
|---|---:|
| Initial BTC Price | 65,000 |
| Knock-Out Ratio | 103% |
| Knock-In Ratio | 75% |
| Annualised Volatility | 50% |
| Dividend Yield Assumption | 6.3% |
| Risk-Free Rate | 1.5% |

The calibrated theoretical coupon is:

```text
47.1111%
```

The substantially larger coupon reflects the much higher volatility assumption in the BTC example.

---

# Greeks & Sensitivity Analysis

The pricing engine is extended into a numerical risk-sensitivity framework.

The project estimates:

- Delta
- Gamma
- Vega
- Theta
- Rho

using finite-difference repricing.

Rather than reporting only a single Greek at inception, several sensitivities are evaluated across a wide range of underlying-price levels.

---

# Delta

Delta measures the sensitivity of product value to the underlying price:

$$
\Delta
=
\frac{\partial V}{\partial S}
$$

The project approximates Delta using a central finite difference:

$$
\Delta
\approx
\frac{
V(S+h)-V(S-h)
}{
2h
}
$$

with:

$$
h = 1\% \times S
$$

The underlying-price grid ranges approximately from:

```text
2,700 to 8,100
```

for the CSI 500 example.

Common random numbers are used between the upward and downward price bumps to reduce Monte Carlo noise in the numerical derivative.

The result is a full:

```text
Delta vs Underlying Price
```

sensitivity curve.

---

# Gamma

Gamma measures the curvature of the product value with respect to the underlying:

$$
\Gamma
=
\frac{\partial^2 V}{\partial S^2}
$$

The numerical approximation is:

$$
\Gamma
\approx
\frac{
V(S+h)-2V(S)+V(S-h)
}{
h^2
}
$$

Gamma is especially relevant for a barrier product because the product's local sensitivity can change materially as the underlying moves toward knock-in or knock-out regions.

---

# Vega

Vega measures sensitivity to the underlying volatility assumption:

$$
\nu
=
\frac{\partial V}{\partial \sigma}
$$

The project reprices the structure after bumping volatility upward and downward by one percentage point:

$$
Vega
\approx
\frac{
V(\sigma+0.01)-V(\sigma-0.01)
}{
0.02
}
$$

The output shows how Snowball valuation changes across different underlying levels when implied / assumed volatility moves.

---

# Theta

Theta measures the sensitivity of the product value to the passage of time:

$$
\Theta
=
\frac{\partial V}{\partial t}
$$

The implementation advances the valuation date by one day and compares the resulting simulated product values.

This provides a numerical approximation of the contract's daily time-decay behaviour.

---

# Rho

Rho measures the sensitivity of the product value to interest rates:

$$
\rho
=
\frac{\partial V}{\partial r}
$$

The risk-free rate is bumped by approximately:

```text
+100 bps / -100 bps
```

and the product is repriced under both scenarios:

$$
\rho
\approx
\frac{
V(r+0.01)-V(r-0.01)
}{
0.02
}
$$

The result is evaluated across the underlying-price grid to produce a dynamic Rho curve.

---

# Path Simulation & Barrier Visualisation

The project also exposes the complete simulated price trajectories rather than returning only the expected product value.

A dedicated path-simulation routine generates:

```text
5,000 Monte Carlo paths
```

and a subset is sampled for visualisation.

The resulting chart overlays:

- Simulated underlying trajectories
- Knock-in barrier
- Knock-out barrier
- Contract horizon

This makes the path dependence of the structure directly observable.

Conceptually:

```text
Underlying Price
      ^
      | ---------------- Knock-Out Barrier
      |      /\/\      /
      |  /\ /    \ /\ /
      | /  \      V  \
      |---------------- Initial Level
      |      \   /
      |-------\-/------- Knock-In Barrier
      |
      +----------------------------> Time
```

Different trajectories therefore illustrate paths that:

- Knock out early
- Survive until maturity
- Breach the knock-in barrier
- Remain between the two barriers

---

# Pathwise Payoff Distribution

Expected product value alone is insufficient for tail-risk analysis.

The stress-testing extension therefore modifies the original valuation logic to return the payoff for **every individual Monte Carlo path**:

$$
\{
Payoff_1,
Payoff_2,
\ldots,
Payoff_M
\}
$$

Loss is then defined as:

$$
Loss_i = -Payoff_i
$$

This creates an empirical loss distribution from which tail-risk measures can be estimated.

---

# Value-at-Risk

The project calculates VaR at a:

```text
97.5% confidence level
```

The loss VaR is defined as:

$$
VaR_{97.5\%}
=
Q_{0.975}(Loss)
$$

where $Q$ denotes the empirical loss quantile.

---

# Expected Shortfall

Expected Shortfall measures the average loss within the tail beyond VaR:

$$
ES_{97.5\%}
=
E
\left[
Loss
\mid
Loss \geq VaR_{97.5\%}
\right]
$$

The stress engine uses:

```text
100,000 Monte Carlo paths per scenario
```

At a 97.5% confidence level, approximately:

```text
2,500 paths
```

fall into the Expected Shortfall tail.

---

# Interest-Rate Stress Testing

Four interest-rate regimes are evaluated.

## Scenario 1 — Normal Rates

```text
Rate Shift:                 0 bps
Rate Volatility Multiplier: 1.0x
```

## Scenario 2 — Rate Up Stress

```text
Rate Shift:                 +200 bps
Rate Volatility Multiplier: 1.0x
```

## Scenario 3 — Severe Rate Stress

```text
Rate Shift:                 +300 bps
Rate Volatility Multiplier: 3.0x
```

## Scenario 4 — Rate Down Stress

```text
Rate Shift:                 -100 bps
Rate Volatility Multiplier: 1.5x
```

Each scenario produces a new pathwise payoff distribution.

---

# Stress-Test Results

| Scenario | 97.5% VaR | 97.5% ES | Mean Payoff | Worst Payoff |
|---|---:|---:|---:|---:|
| Normal Rates | **0.3818** | **0.4240** | 0.0370 | -0.5989 |
| Rate Up +200 bps | **0.3429** | **0.3854** | 0.0581 | -0.5605 |
| +300 bps + Higher Rate Vol | **0.3244** | **0.3690** | 0.0666 | -0.5551 |
| Rate Down -100 bps | **0.4016** | **0.4444** | 0.0240 | -0.6224 |

The downside-rate scenario produces the largest simulated tail loss:

```text
97.5% ES = 0.4444
Worst Path Payoff = -0.6224
```

By comparison, the severe positive-rate scenario produces:

```text
97.5% ES = 0.3690
Mean Payoff = 0.0666
```

Within this parameterisation, downward rate shocks therefore generate the most adverse simulated tail-risk profile among the tested interest-rate scenarios.

---

# Stress Distribution Analysis

The notebook compares the full loss distributions under all rate scenarios.

The research output includes:

```text
Normal Rate Loss Distribution
        vs
+200 bps Rate Shock
        vs
+300 bps + Higher Rate Volatility
        vs
-100 bps Rate Shock
```

Expected Shortfall is also plotted across scenarios to compare extreme-loss exposure directly.

---

# Key Results

## Pricing

```text
CSI 500 Expected Payoff:          ~0.03740
CSI 500 Theoretical Coupon:       12.7064%
BTC Theoretical Coupon:           47.1111%
Monte Carlo Pricing Paths:        300,000
```

## Sensitivity Analysis

```text
Greeks:
- Delta
- Gamma
- Vega
- Theta
- Rho

Underlying sensitivity range:
~2,700 to ~8,100
```

## Tail Risk

```text
Stress Simulation Paths:          100,000 per scenario
Tail Confidence Level:            97.5%
Normal ES:                        0.4240
Rate-Down Stress ES:              0.4444
Rate-Down Worst Payoff:          -0.6224
```

---

# Technology Stack

```text
Python
│
├── PyTorch
│   ├── Monte Carlo path generation
│   ├── Tensor-based payoff calculation
│   └── CPU / GPU device support
│
├── NumPy
│   └── Numerical calculations
│
├── pandas
│   ├── Contract dates
│   └── Monitoring schedule management
│
├── pandas-market-calendars
│   └── Shanghai Stock Exchange trading calendar
│
├── Matplotlib
│   ├── Greek sensitivity curves
│   ├── Simulated price paths
│   ├── Loss distributions
│   └── Expected Shortfall comparison
│
└── tqdm
    └── Progress tracking for repeated repricing
```

---

# Project Architecture

The project can be viewed as three connected quantitative modules.

## 1. Pricing Engine

```text
Contract Parameters
      |
      v
Trading Calendar
      |
      v
GBM + Hull–White Simulation
      |
      v
KI / KO Payoff Evaluation
      |
      v
Monte Carlo Expected Value
      |
      v
Coupon Calibration
```

---

## 2. Sensitivity Engine

```text
Base Monte Carlo Value
      |
      +--> Spot Bumps --> Delta / Gamma
      |
      +--> Volatility Bumps --> Vega
      |
      +--> Date Shift --> Theta
      |
      +--> Rate Bumps --> Rho
```

---

## 3. Tail-Risk Engine

```text
Pathwise Payoffs
      |
      v
Loss Distribution
      |
      v
97.5% VaR
      |
      v
97.5% Expected Shortfall
      |
      v
Rate Stress Scenarios
```

---

# Research Scope

This repository is a numerical research framework for **structured-product pricing and risk analytics**.

It connects three stages of quantitative analysis:

```text
Valuation
    ↓
Sensitivity
    ↓
Tail Risk
```

The project therefore goes beyond producing a single Snowball fair value.

It studies how the same path-dependent product behaves under:

- Changes in the underlying price
- Changes in volatility
- Passage of time
- Changes in interest rates
- Stochastic rate paths
- Extreme tail-loss scenarios

The framework is intended to demonstrate the relationship between **Monte Carlo valuation, nonlinear barrier exposure and portfolio risk measurement** in structured derivatives.