## Data generation

#### Overall specifications & constraints

The main purpose of this document is to define the specification and constraints for the data generation phase of a ***Machine Learning Device Positioning Prediction Model.***

We are emulating ***2D indoor and outdoor environments*** with multiple devices operating within a ***5 GHz frequency band***. To avoid bias, both scenarios are assigned an equal probability of being generated during the data generation phase.

**Simplified data generation process:**

1. Define the environment type (*indoor or outdoor*) based on a random selection process.
2. Generate network characteristics given the environment type.
3. Return the generated data in a structured format for positioning estimation calculations.

<br>

### Grid

- **Dimensionality:**  
  The environment is modelled as a two-dimensional Cartesian coordinate space, with positions represented as $(x, y)$ coordinate pairs.

- **Restricted domain and range $(x, y)$:**  
  $-10 < x < 10,\ -10 < y < 10$  
  representing an indoor or outdoor environment (**20 m × 20 m = 400 m²**), such as a shopping mall, field, or warehouse.

- **Limited device generation:**  
  The number of devices in the network is constrained to a manageable range (e.g. **15–30**) to ensure the dataset remains tractable for training while still providing sufficient geometric coverage for accurate positioning.

<br>

---

### Time Difference of Arrival (TDOA)

TDOA measures the **difference in signal arrival times** between pairs of receivers and relates it to the **difference in propagation distances** divided by the signal propagation speed:

$$
\Delta t_i = \frac{(d_i - d_0)}{c}
$$

> **Difference between (target → A) and (target → B) distances gives TDOA**

One device is designated as the **target transmitter**. All remaining devices act as **synchronised anchors** and measure the arrival time of the target’s transmission.

Time Difference of Arrival (TDOA) measurements are formed by **differencing arrival times at anchor pairs**.

<br>

---

### Position Estimation Parameters

| Parameter | Symbol / Formula | Description |
|----------|------------------|-------------|
| Anchor positions | $(x_i, y_i)$ | Cartesian coordinates of receiver (anchor) devices with known locations |
| Target position (ground truth) | $(x_T, y_T)$ | True Cartesian coordinates of the target device (used for data generation and evaluation) |
| Signal propagation speed | $c = 3 \times 10^8\ \text{m/s}$ | Assumed constant RF propagation speed |
| Number of anchors | $N \geq 2$ | Minimum of two anchors required to form a TDOA measurement |
| Target point | $P$ | Geometric point representing the target device location |
| Anchor points | $A_0, A_1, \ldots, A_N$ | Set of anchor device locations |
| Target–anchor distance | $d_i = \|P - A_i\|$ | Euclidean distance between target and anchor $A_i$ |
| Reference distance | $d_0 = \|P - A_0\|$ | Distance between the target and the reference anchor |
| Ideal TDOA | $\Delta t_i = \frac{(d_i - d_0)}{c}$ | Noise-free time difference of arrival relative to reference anchor $A_0$ |
| Timing noise / bias | $\epsilon_i$ | Random or systematic timing error added to emulate realistic measurements |
| Noisy TDOA | $\Delta t_i^{\text{obs}} = \Delta t_i + \epsilon_i$ | Observed TDOA used for positioning and ML training. See this [paper on Gaussian noise models for TOA and TDOA](https://ieeexplore.ieee.org/abstract/document/6289832) for details |

<br>

---

### Conceptual Example

#### Roles

- **T** → Target device (transmitter, unknown position)
- **A, B, C, …** → Anchors / reference devices (receivers, known positions, synchronised clocks)

<br>

#### Calculations

1. **T transmits** a signal once.
2. **A, B, C** receive the *same* transmission.
3. Each anchor records an arrival time:
   - Using $\Delta t_i = \frac{(d_i - d_0)}{c}$
   - Arrival times: $t_A,\ t_B,\ t_C$
4. Select a reference anchor **A**.
5. Form TDOA measurements by differencing arrival times:

- **TDOA(B − A):** $t_B - t_A$  
- **TDOA(C − A):** $t_C - t_A$

<br>

---

### Visual Example

```javascript
const network = {
  environment: {
    type: "indoor",
    propagationSpeed: 3e8,       // c (m/s)
    timingNoiseStdDev: 0.2       // σ_t in ns (scenario-level)
  },

  devices: {
    targetDevice: {
      id: "T",
      position: [3.5, -2.0]      // Unknown during inference
    },

    // anchors
    otherDevices: {
      A: { position: [-6.0, 4.5], isReference: true },
      B: { position: [8.0, 1.0],  isReference: false },
      C: { position: [-2.0, -7.5], isReference: false }
    }
  },

  channels: {
    "T-A": {
      distance: 11.2,            // meters
      arrivalTime: 37.33,        // ns (d / c + noise)
      timingNoise: -0.05         // ns
    },

    "T-B": {
      distance: 6.8,
      arrivalTime: 22.69,
      timingNoise: 0.12
    },

    "T-C": {
      distance: 5.9,
      arrivalTime: 19.79,
      timingNoise: 0.08
    }
  },

  tdoaMeasurements: {
    referenceAnchor: "A",

    "B-A": {
      deltaDistance: -4.4,       // d_B - d_A (meters)
      tdoa: -14.64,              // t_B - t_A (ns)
      noisyTdoa: -14.47          // with timing noise
    },

    "C-A": {
      deltaDistance: -5.3,
      tdoa: -17.54,
      noisyTdoa: -17.41
    }
  }
};

```
