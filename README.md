# Snowball Pricing & Risk Analytics with Monte Carlo and Hull–White

## Overview

This project develops a numerical framework for pricing and analysing path-dependent **Snowball / ABRC structured products**.

The framework combines Monte Carlo simulation, barrier-event logic, stochastic interest-rate modelling, coupon calibration, Greeks, path visualisation, and tail-risk stress testing.

The project is organised around three main layers:

```text
1. Pricing
   Monte Carlo paths
   -> Knock-in / knock-out payoff logic
   -> Expected payoff
   -> Theoretical coupon calibration

2. Sensitivity Analysis
   Spot / volatility / time / rate shocks
   -> Delta
   -> Gamma
   -> Vega
   -> Theta
   -> Rho

3. Risk Analytics
   Pathwise payoff distribution
   -> VaR
   -> Expected Shortfall
   -> Interest-rate stress scenarios
```

---

## Project Workflow

```text
Contract Terms
     |
     v
Trading Calendar & Observation Dates
     |
     v
Underlying Price Simulation
     +
Hull-White Rate Simulation
     |
     v
Knock-In / Knock-Out Path Classification
     |
     v
Monte Carlo Valuation
     |
     v
Theoretical Coupon Calibration
     |
     v
Greeks & Sensitivity Curves
     |
     v
Pathwise Payoff Distribution
     |
     v
VaR / Expected Shortfall
     |
     v
Stress-Scenario Analysis
```

---

## 1. Snowball Contract Structure

The Snowball is a path-dependent structured product whose payoff depends on whether the underlying asset reaches predefined barriers during the life of the contract.

The main contract inputs include:

- Initial underlying price
- Valuation price
- Contract start and maturity dates
- Knock-in barrier
- Knock-out barrier
- Knock-out monitoring dates
- Coupon rate
- Underlying volatility
- Dividend yield
- Risk-free rate
- Notional / margin ratio

A baseline CSI 500-linked Snowball uses:

| Parameter | Value |
|---|---:|
| Initial Price | 5,400 |
| Knock-Out Barrier | 103% of initial price |
| Knock-In Barrier | 75% of initial price |
| Annual Volatility | 14% |
| Dividend Yield | 6.3% |
| Risk-Free Rate | 1.5% |
| Contract Start | 2024-01-02 |
| Maturity | 2025-12-31 |
| Knock-Out Observations | 21 dates |

Barrier levels are therefore:

```text
Knock-Out Level = 1.03 x 5,400
Knock-In Level  = 0.75 x 5,400
```

---

## 2. Trading Calendar

The pricing engine uses the Shanghai Stock Exchange calendar through `pandas_market_calendars`.

The calendar is used to:

- Remove non-trading days
- Construct the simulation grid
- Align knock-out observation dates
- Determine the remaining life of the contract from the valuation date

This is important because Snowball payoff conditions are evaluated over actual market trading dates rather than a simple calendar-day grid.

---

## 3. Monte Carlo Underlying Simulation

Underlying prices are simulated using a Geometric Brownian Motion process.

The implemented update can be written as:

```text
S(t+dt)
=
S(t) x exp[
    (r - q - 0.5 x sigma^2) x dt
    + sigma x sqrt(dt) x Z
]
```

where:

```text
S      = underlying price
r      = risk-free rate
q      = dividend yield
sigma  = underlying volatility
dt     = time step
Z      = standard normal random shock
```

The main valuation engine uses:

```text
300,000 Monte Carlo paths
```

PyTorch tensors are used for numerical simulation and payoff calculations.

The execution device is automatically selected:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

PyTorch is used here as a high-performance numerical computing framework rather than for deep learning.

---

## 4. Hull–White Interest-Rate Simulation

The framework additionally models stochastic short-rate paths using a mean-reverting Hull–White-style process.

The implemented rate evolution is approximately:

```text
r(t+dt)
=
r(t)
+ a x [r0 - r(t)] x dt
+ sigma_r x sqrt(dt) x Z
```

Baseline parameters:

```text
Mean-Reversion Speed:  a       = 0.10
Rate Volatility:       sigma_r = 0.01
Initial Rate:          r0      = 1.5%
```

Each simulated short-rate path is converted into a path-dependent discount factor.

Conceptually:

```text
Discount Factor
=
exp(
    - cumulative sum of simulated rates x dt
)
```

This allows cash flows from knock-out or maturity events to be discounted under simulated rate paths rather than relying only on a single fixed discount rate.

---

## 5. Path-Dependent Payoff Logic

Each simulated underlying path is classified according to its barrier history.

### Knock-Out

On every scheduled knock-out observation date, the model checks whether the underlying exceeds the knock-out barrier.

```text
If price >= Knock-Out Barrier:
    identify the first qualifying observation date
    calculate accrued coupon
    discount the payoff
    terminate the contract
```

The earliest knock-out date determines the payoff timing.

### No Knock-Out and No Knock-In

If the product never knocks out and never breaches the knock-in barrier, the investor receives the maturity coupon.

### Knock-In Without Knock-Out

If the product breaches the knock-in barrier but does not subsequently knock out, the maturity payoff may become linked to the terminal performance of the underlying.

The downside component is represented as:

```text
Terminal Return
=
Final Underlying Price / Initial Underlying Price - 1
```

The Monte Carlo pricing engine applies these payoff rules separately to every simulated path.

---

## 6. Monte Carlo Valuation

Once all path-dependent outcomes have been evaluated, the Snowball value is estimated by averaging discounted payoffs across simulated paths.

```text
Monte Carlo Value
=
Average of discounted payoff across all simulated paths
```

This framework therefore captures:

- Early knock-out
- Knock-in exposure
- Survival to maturity
- Time-dependent coupon accrual
- Stochastic underlying paths
- Stochastic discount factors

---

## 7. Theoretical Coupon Calibration

The project also solves the inverse pricing problem:

> What coupon rate makes the simulated Snowball value equal to the target contract value?

A binary-search procedure repeatedly reprices the structure under different coupon assumptions.

```text
Choose Coupon
     |
     v
Run Monte Carlo Pricing
     |
     v
Compare Value with Target
     |
     +---- Too High -> Reduce Coupon
     |
     +---- Too Low  -> Increase Coupon
     |
     v
Repeat Until Convergence
```

The search tolerance is approximately:

```text
1e-5
```

### CSI 500 Result

The calibrated theoretical coupon is:

```text
12.7064%
```

### BTC Snowball Example

The same framework is also applied to a BTC-linked Snowball with:

```text
Initial BTC Price:   65,000
Volatility:          50%
Knock-In Ratio:      75%
Knock-Out Ratio:     103%
```

The resulting theoretical coupon is:

```text
47.1111%
```

The higher coupon reflects the substantially larger volatility assumption in the BTC-linked example.

---

# Greeks & Sensitivity Analysis

The Monte Carlo pricer is extended into a numerical sensitivity engine.

The project calculates:

```text
Delta
Gamma
Vega
Theta
Rho
```

using repeated repricing under controlled parameter shocks.

---

## 8. Delta

Delta measures how the Snowball value changes as the underlying price changes.

The central-difference approximation is:

```text
Delta
≈
[V(S + h) - V(S - h)] / (2h)
```

with:

```text
h = 1% of the underlying price
```

For the CSI 500 analysis, the valuation price is scanned over a broad range:

```text
Approximately 2,700 to 8,100
```

This produces a full Delta curve instead of only a single point estimate.

The upward and downward price bumps reuse the same Monte Carlo shocks where applicable, reducing simulation noise in the finite-difference estimate.

---

## 9. Gamma

Gamma measures the curvature of Snowball value with respect to the underlying.

It is approximated by:

```text
Gamma
≈
[V(S + h) - 2V(S) + V(S - h)] / h^2
```

The resulting Gamma curve shows how convexity changes as the underlying approaches different barrier regions.

This is particularly relevant for Snowballs because their risk profile can change sharply near knock-in and knock-out levels.

---

## 10. Vega

Vega measures sensitivity to underlying volatility.

The model reprices the Snowball after increasing and decreasing volatility by one percentage point:

```text
Vega
≈
[V(sigma + 0.01) - V(sigma - 0.01)] / 0.02
```

This allows the project to examine how the value of the path-dependent structure changes under different volatility assumptions.

---

## 11. Theta

Theta measures the effect of the passage of time.

The implementation moves the valuation date forward and reprices the product.

Conceptually:

```text
Theta
≈
Value one day later - Current value
```

The calculation captures changes caused by:

- Shorter remaining maturity
- Fewer remaining observation opportunities
- Changing barrier-event probabilities
- Time decay of future coupon exposure

---

## 12. Rho

Rho measures sensitivity to changes in interest rates.

The risk-free rate is shocked upward and downward and the product is repriced.

```text
Rho
≈
[V(r + 0.01) - V(r - 0.01)] / 0.02
```

The project evaluates Rho across different underlying-price levels rather than reporting only a single inception value.

---

# 13. Monte Carlo Path Visualisation

The pricing framework can also return complete simulated underlying trajectories.

A separate simulation generates:

```text
5,000 paths
```

with a smaller subset displayed for visual analysis.

The chart overlays:

```text
Simulated Underlying Paths
+
Knock-Out Barrier
+
Knock-In Barrier
```

This makes the path-dependent structure directly visible.

Typical simulated outcomes include:

```text
Path A -> Crosses Knock-Out Barrier -> Early Redemption

Path B -> Remains Between Barriers -> Survives to Maturity

Path C -> Crosses Knock-In Barrier -> Downside Exposure

Path D -> Knock-In First, Then Recovers Toward Maturity
```

---

# Tail-Risk Analytics

The final section moves from expected valuation to the full distribution of possible contract outcomes.

---

## 14. Pathwise Payoff Distribution

Standard Monte Carlo valuation returns the average payoff.

Tail-risk measurement instead requires the payoff generated by every simulated path.

The framework therefore constructs:

```text
Path 1   -> Payoff 1
Path 2   -> Payoff 2
Path 3   -> Payoff 3
...
Path M   -> Payoff M
```

Loss is defined as:

```text
Loss = -Pathwise Payoff
```

The resulting empirical loss distribution is then used for VaR and Expected Shortfall calculations.

---

## 15. Value-at-Risk

The project calculates Value-at-Risk at a:

```text
97.5% confidence level
```

Conceptually:

```text
97.5% VaR
=
97.5th percentile of simulated losses
```

This represents the loss threshold exceeded by the worst 2.5% of simulated outcomes.

---

## 16. Expected Shortfall

Expected Shortfall measures the average loss conditional on being inside the VaR tail.

```text
97.5% Expected Shortfall
=
Average loss among paths
whose loss exceeds the 97.5% VaR threshold
```

Each stress scenario uses:

```text
100,000 Monte Carlo paths
```

At the 97.5% confidence level:

```text
Worst 2.5% of paths
=
2,500 simulated observations
```

---

# 17. Interest-Rate Stress Testing

The tail-risk engine compares four interest-rate environments.

### Normal Rates

```text
Rate Shift:                 0 bps
Rate-Volatility Multiplier: 1.0x
```

### Rate Up Stress

```text
Rate Shift:                 +200 bps
Rate-Volatility Multiplier: 1.0x
```

### Severe Rate Stress

```text
Rate Shift:                 +300 bps
Rate-Volatility Multiplier: 3.0x
```

### Rate Down Stress

```text
Rate Shift:                 -100 bps
Rate-Volatility Multiplier: 1.5x
```

Each scenario generates an independent 100,000-path payoff distribution.

---

## 18. Stress-Test Results

| Scenario | 97.5% VaR | 97.5% ES | Mean Payoff | Worst Payoff | Best Payoff |
|---|---:|---:|---:|---:|---:|
| Normal Rates | 0.3818 | 0.4240 | 0.0370 | -0.5989 | 0.4076 |
| Rate Up +200 bps | 0.3429 | 0.3854 | 0.0581 | -0.5605 | 0.3899 |
| Severe +300 bps / Higher Rate Vol | 0.3244 | 0.3690 | 0.0666 | -0.5551 | 0.4106 |
| Rate Down -100 bps | 0.4016 | 0.4444 | 0.0240 | -0.6224 | 0.4263 |

Each scenario contains:

```text
Expected Shortfall Tail Observations: 2,500
```

---

## Stress Interpretation

Within the tested parameter set, the rate-down scenario produces the most adverse tail-risk profile.

```text
Normal Rates
97.5% ES = 0.4240

Rate Down Stress
97.5% ES = 0.4444
```

The worst simulated payoff also deteriorates from:

```text
Normal:     -0.5989
Rate Down:  -0.6224
```

By comparison, the severe rate-up / high-rate-volatility scenario produces:

```text
97.5% VaR = 0.3244
97.5% ES  = 0.3690
Mean Payoff = 0.0666
```

The stress-testing module therefore allows the impact of rate level and rate volatility to be compared directly through the full simulated payoff distribution.

---

# Key Outputs

## Pricing

```text
Monte Carlo Valuation Paths:       300,000

CSI 500 Theoretical Coupon:        12.7064%

BTC Theoretical Coupon:            47.1111%
```

## Risk Sensitivities

```text
Delta
Gamma
Vega
Theta
Rho
```

evaluated through repeated Monte Carlo repricing.

## Tail Risk

```text
Stress Paths per Scenario:         100,000

Confidence Level:                  97.5%

Normal VaR:                        0.3818
Normal ES:                         0.4240

Rate-Down Stress VaR:              0.4016
Rate-Down Stress ES:               0.4444

Worst Rate-Down Payoff:           -0.6224
```

---

# Technology Stack

```text
Python
|
|-- PyTorch
|   |-- Monte Carlo simulation
|   |-- Vectorised payoff calculations
|   |-- CPU / GPU execution
|
|-- NumPy
|   |-- Numerical calculations
|
|-- pandas
|   |-- Contract-date processing
|   |-- Monitoring schedules
|
|-- pandas-market-calendars
|   |-- Shanghai Stock Exchange calendar
|
|-- Matplotlib
|   |-- Greeks curves
|   |-- Simulated price paths
|   |-- Loss distributions
|   |-- Stress comparisons
|
|-- tqdm
    |-- Progress tracking for repeated simulations
```

---

# Project Structure

The research framework can be viewed as three connected modules.

```text
PRICING ENGINE
Contract Terms
    |
    v
GBM + Hull-White Simulation
    |
    v
Barrier Logic
    |
    v
Expected Payoff
    |
    v
Coupon Calibration


SENSITIVITY ENGINE
Base Monte Carlo Value
    |
    +--> Spot Shock ------> Delta / Gamma
    |
    +--> Vol Shock -------> Vega
    |
    +--> Time Shift ------> Theta
    |
    +--> Rate Shock ------> Rho


RISK ENGINE
Pathwise Payoffs
    |
    v
Loss Distribution
    |
    v
VaR
    |
    v
Expected Shortfall
    |
    v
Rate Stress Scenarios
```

---

# Summary

This project builds a quantitative research framework for analysing the nonlinear risk profile of Snowball structured products.

It connects three parts of derivatives analysis:

```text
Monte Carlo Pricing
        |
        v
Risk Sensitivities
        |
        v
Tail-Risk Stress Testing
```

The framework supports:

- Path-dependent knock-in / knock-out valuation
- Stochastic underlying simulation
- Hull–White short-rate simulation
- Theoretical coupon calibration
- Numerical Greeks
- Barrier-path visualisation
- Pathwise payoff analysis
- 97.5% VaR
- 97.5% Expected Shortfall
- Interest-rate stress testing

The result is a single workflow for studying how Snowball value and risk change across underlying prices, volatility assumptions, time, interest rates, and extreme market scenarios.