# Crypto Risk Index: Mathematical Foundations

This document provides the theoretical background for the four indices implemented in **crypto-risk-index**:

1. Tail Fragility Index (TFI)
2. Order-Flow Toxicity Index (OTI)
3. Leverage Fragility Index (LFI)
4. Directional Jump Risk (JRI)

> **Important:** TFI, OTI, LFI and the JRI composition are research diagnostics. Except for concepts derived from published literature (large deviations, VPIN, perpetual futures mechanics), the specific composite indices implemented in this package are original engineering constructs and should be validated empirically before being interpreted as predictive models.

---

# 1. Tail Fragility Index (TFI)

## Motivation

Rather than estimating volatility, TFI estimates how **easy or difficult** it is for the empirical return distribution to generate large moves.

Returns are

$$
r_t=\log\bigl(P_t / P_{t-\Delta}\bigr).
$$

Robustly standardized returns are

$$
z_t=
\frac{r_t-m_t}{s_t+\varepsilon},
$$

where the scale is preferably the median absolute deviation.

The empirical cumulant generating function is

$$
\widehat{\Lambda}(\theta)=
\log\bigl((1/n)\sum_i e^{\theta z_i}\bigr)
$$

The empirical rate function is obtained using the Legendre–Fenchel transform

$$
\widehat{I}(x)=
\sup_{\theta}\bigl(\theta x-\widehat{\Lambda}(\theta)\bigr)
$$

Downside and upside tail costs are

$$
C^-_k=I(-k),
$$

$$
C^+_k=I(+k).
$$

Low values indicate a weak statistical barrier for large moves.

Rate-surface collapse is

$$
Collapse^-=
I_{slow}(-3)-I_{fast}(-3),
$$

with the mirrored definition for the upside.

Asymmetry is

$$
A_3=I(+3)-I(-3).
$$

Velocity is

$$
V^-=
-(I_t(-3)-I_{t-h}(-3)).
$$

The package combines normalized tail cost, collapse, asymmetry and velocity into a directional fragility score.

### Interpretation

TFI measures **distributional susceptibility**, not probability.

---

Recommended reading

- Hugo Touchette, *A Basic Introduction to Large Deviations*.
- Dembo & Zeitouni, *Large Deviations Techniques and Applications*.
- Varadhan, *Large Deviations*.

---

# 2. Order-Flow Toxicity Index (OTI)

OTI measures whether aggressive trading is overwhelming available liquidity.

Classical VPIN computes

$$
VPIN
=
\frac{1}{N}
\sum
\frac{|V^B-V^S|}{V_B}.
$$

The package extends this into directional measures.

Buy toxicity

$$
BuyTox=
\frac{1}{N}
\sum
\max(0,S_b),
$$

Sell toxicity

$$
SellTox=
\frac{1}{N}
\sum
\max(0,-S_b).
$$

Liquidity pressure compares aggressive volume to resting depth

$$
SellPressure
=
\frac{AggSell}{BidDepth}.
$$

Bid depletion

$$
BidDepletion=
\frac{Depth_{old}-Depth_{new}}{Depth_{old}}.
$$

Spread stress

$$
Spread=
\frac{Ask-Bid}{Mid}.
$$

The downside OTI combines sell toxicity, sell pressure, bid depletion, replenishment failure and spread stress.

The upside OTI mirrors the calculation.

### Interpretation

OTI measures **active market stress** rather than structural fragility.

---

Recommended reading

- Easley, López de Prado & O'Hara, *Flow Toxicity and Liquidity in a High-Frequency World*.
- *The Volume Clock*.
- Scaillet et al., *High Frequency Jump Analysis of Bitcoin*.

---

# 3. Leverage Fragility Index (LFI)

LFI measures whether derivatives positioning is crowded and beginning to unwind.

Funding is normalized

$$
F^{8h}
=
F
\frac{8}{h}.
$$

Basis

$$
Basis=
\frac{Perp-Spot}{Spot}.
$$

Open interest change

$$
\Delta OI=
\frac{OI_t-OI_{t-h}}{OI_{t-h}}.
$$

Leverage density

$$
OIToVolume=
\frac{OI}{Volume_{24h}}.
$$

Long crowding

$$
LC^-=
w_1F+w_2Basis+w_3\Delta OI+w_4OIToVolume.
$$

Short crowding mirrors the signs.

Long unwind combines

- falling price
- falling basis
- falling OI
- long liquidations.

Finally

$$
LFI^-=
0.45C^-+
0.35U^-+
0.20C^-U^-.
$$

with the mirrored upside score.

### Interpretation

Crowding and unwind are intentionally separated.

A high funding rate alone is **not** a crash signal.

---

Recommended reading

- He et al., *Fundamentals of Perpetual Futures*.
- Cheng et al., *Liquidation, Leverage and Optimal Margin in Bitcoin Futures Markets*.
- OKX and Bybit documentation on funding, mark price and open interest.

---

# 4. Directional Jump Risk

The package composes the previous layers

$$
J^-=
w_TTFI^-+
w_LLFI^-+
w_OOTI^-+
w_ITFI^-LFI^-OTI^-.
$$

The upside score is analogous.

This is a ranking score rather than a calibrated probability.

To estimate probabilities, fit models using future MAE/MFE labels with walk-forward validation.

---

## References

- Touchette (2011) Large Deviations.
- Easley, López de Prado & O'Hara (VPIN).
- He et al. Fundamentals of Perpetual Futures.
- OKX API and Funding documentation.
- Bybit Funding and Mark Price documentation.
