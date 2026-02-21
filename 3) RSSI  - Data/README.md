# Data generation

#### Overall specifications & constraints

The main purpose of this document is to define the specification and constraints for the data generation phase of a ***Machine Learning Device Positioning Prediction Model.***

We are emulating ***2D indoor and outdoor environments*** with multiple devices operating within a ***5 GHz frequency band***. To avoid bias, we ensure that both scenarios have an equal probability of being generated during data generation.

**Simplified data generation process:**
1. Define the environment type (*indoor or outdoor*) using a random selection process.
2. Generate network characteristics based on the environment type.
3. Return the generated data in a structured format for positioning estimation calculations.

<br>

### Grid

- **Dimensionality:**  
  The environment is modelled as a two-dimensional Cartesian coordinate space, with positions represented as $(x, y)$ coordinate pairs.

- **Restricted domain and range $(x, y)$:**  
  $-30 < x < 30,\ -30 < y < 30$  
  representing an indoor or outdoor environment (**60 m × 60 m = 3600 m²**), such as a shopping mall, field, or warehouse.

- **Limited device generation:**  
  The number of devices/endpoints in the network is constrained to a manageable range (e.g. **15–30**) to ensure the dataset remains tractable for training while still providing sufficient geometric coverage for accurate positioning.
  The number of antennas will be 5-10.

<br>

## RSSI Calculation

### Path Loss Exponent (Environment-Specific)

The path loss exponent $n$ characterises the average rate at which signal power decays with distance and depends on the propagation environment.

In the data generation process, $n$ is sampled once per scenario and remains fixed for all links within that scenario to preserve spatial consistency.

We will only use outdoor, Indoor LOS and Indoor NLOS environment types. 

**Assumed values used in this project:**

| Environment Type | Description | Path Loss Exponent $n$ |
|------------------|------------|------------------------|
| Free Space / Open Outdoor | Minimal obstruction, near line-of-sight | 2.0 – 2.2 |
| Outdoor (Urban / Light Clutter) | Buildings, foliage | 2.7 – 3.5 |
| Indoor LOS | Open indoor areas (e.g. halls, warehouses) | 2.0 – 3.0 |
| Indoor NLOS | Multiple walls, partitions | 3.0 – 6.0 |
| Heavily Obstructed Indoor | Dense materials (concrete, metal) | 4.0 – 6.0 |

> **Reference:**  
> Srinivasan, S.; Haenggi, M., *Path loss exponent estimation in large wireless networks*,  
> Information Theory and Applications Workshop, Feb 2009.  
> https://arxiv.org/pdf/0802.0351v1.pdf

<br>

---

### Log-Distance Path Loss Model

The large-scale average path loss between two devices separated by distance $d$ is modelled using the **log-distance path loss model**:

$$
PL(d) = PL(d_0) + 10 n \log_{10}\!\left(\frac{d}{d_0}\right)
$$

where:
- $PL(d)$ is the path loss at distance $d$ (dB)
- $d_0$ is a reference distance (typically $1\,\text{m}$)
- $n$ is the path loss exponent
- $PL(d_0)$ is the free-space path loss at $d_0$

The free-space path loss at the reference distance is given by:

$$
PL(d_0) = 20 \log_{10}\!\left(\frac{4\pi d_0}{\lambda}\right)
$$

with wavelength:

$$
\lambda = \frac{c}{f}
$$

where $c$ is the speed of light and $f$ is the carrier frequency (5 GHz).

<br>

---


### Log-Normal Shadowing (Gaussian Noise)

To model environmental variability and measurement uncertainty, log-normal shadowing is applied by adding a **zero-mean Gaussian random variable** $X_\sigma$ in the logarithmic domain:

$$
PL(d) = PL(d_0) + 10 n \log_{10}\!\left(\frac{d}{d_0}\right) + X_\sigma
$$

where:

$$
X_\sigma \sim \mathcal{N}(0,\sigma^2)
$$

The parameter $\sigma$ represents the standard deviation of shadow fading (in dB) and is environment-dependent.

**Typical values used in this project:**

| Environment | $\sigma$ (dB) |
|------------|---------------|
| Free space / open outdoor | 1 – 2 |
| Indoor LOS | 2 – 4 |
| Indoor NLOS | 4 – 8 |
| Heavy obstruction | 6 – 10 |

The Gaussian noise term is applied independently to each wireless link, while $\sigma$ remains fixed per scenario.

<br>

---


### Estimating $n$ and $\sigma$ from Received-Power Samples (MMSE + SSE)

Assume $k$ received power measurements $\{p_i\}_{i=1}^{k}$ 

collected at distances $\{d_i\}_{i=1}^{k}$ from a transmitter, with reference distance $d_0$ and reference power $p(d_0)$.

<br>

#### 1) Log-distance received-power model

$$
\hat{p}_i = p(d_0) - 10n\log_{10}\!\left(\frac{d_i}{d_0}\right)
$$

#### 2) Sum of Squared Errors (SSE)

$$
J(n)=\sum_{i=1}^{k}\left(p_i-\hat{p}_i\right)^2
$$

#### 3) MMSE estimate of $n$

$$
\hat{n}=\arg\min_n J(n)
$$

#### 4) Estimating $\sigma$

$$
\sigma^2 = \frac{J(\hat{n})}{k}, \quad
\sigma = \sqrt{\frac{J(\hat{n})}{k}}
$$

<br>

---


### Using $\sigma$ During Data Generation

For each scenario:
1. Fit $\hat{n}$ using MMSE.
2. Compute $\sigma = \sqrt{J(\hat{n})/k}$.
3. For each wireless link $(i \rightarrow j)$, sample:

$$
X_{\sigma,i,j} = \sigma z,\quad z \sim \mathcal{N}(0,1)
$$

4. Compute received power:

$$
p_r(d) = p(d_0) - 10\hat{n}\log_{10}\!\left(\frac{d}{d_0}\right) + X_{\sigma,i,j}
$$

This ensures:
- $\sigma$ is **fixed per scenario**
- $X_{\sigma,i,j}$ is **independent per link**

<br>

### Calculating RSSI

$$
RSSI_{i,j} = P_t + G_{Tx} + G_{Rx} - PL_{i,j}
$$

where:
- $P_t$ is transmit power (dBm)
- $G_{Tx}$ and $G_{Rx}$ are antenna gains (dBi)
- $PL_{i,j}$ is the total path loss (dB)

<br>

### Visual Example

```javascript
const network = {
  environment: {
    type: "indoor",
    losModel: "NLOS",
    pathLossExponent: 4.2,
    shadowingStdDev: 6.0
  },

  devices: {
    targetDevice: {
      id: "T",
      position: [3.5, -2.0]
    },

    otherDevices: {
      A: { position: [-6.0, 4.5] },
      B: { position: [8.0, 1.0] },
      C: { position: [-2.0, -7.5] },
      D: { position: [5.0, -6.0] }
    }
  },

  channels: {
    "T-A": {
      distance: 11.2,
      freeSpacePathLoss: 65.8,
      shadowing: 2.3,
      pathLoss: 75.5,
      receivedSignalStrength: -62.3
    }
  }
};
```
