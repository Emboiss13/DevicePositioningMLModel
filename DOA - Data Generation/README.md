#  Data generation 

### Overall specifications & constraints
The main purpose of this document is to define the specification and constraints for the data generation phase of a ***Machine Learning Device Positioning Prediction Model.***
We are emulating ***2D indoor and outdoor environments*** with multiple devices operating within a ***5GHz frequency band***. To avoid bias, we will ensure that in the data generation phase both scenarios have an equal chance of being generated.
  
**Simplified data generation process:**

1. Define the environment type (*indoor or outdoor*) based on a random selection process.
2. Generate network characteristics given the environment type.
3. Return the generated data in a structured format for positioning estimation calculations.

<br>

##  Grid

-  **Dimensionality:** The environment is modelled as a two-dimensional Cartesian coordinate space, with positions represented as $\text{(x,y)}$ coordinate pairs.
-  **Restricted Domain and Range (x,y):** $[-10 < x < 10, -10 < y < 10]$ representing an indoor or outdoor environment (20m x 20m = 400 m²), such as a shopping mall, field, or warehouse.
-  **Limited device generation:** We will constrain the number of devices in the network to a manageable number (e.g., 15-30) to ensure the dataset is not too large for training while still providing sufficient coverage for accurate positioning.

<br>

##  Angle of Arrival (AOA) / Direction of Arrival (DOA)
Angle of Arrival (AOA) estimates the **direction from which a signal arrives** at a receiver by analysing **phase differences** across multiple spatially separated antenna elements.
Direction of Arrival (DOA) refers to the **signal processing estimation process**, while AOA denotes the **resulting geometric angle** used for positioning.

In this project, AOA estimates are generated using **array-based DOA estimation**, assuming a **uniform linear antenna array (ULA)** at each anchor device. The estimated AOA provides a **directional constraint** rather than a distance measurement, and therefore must be combined with additional anchors or positioning methods to determine the target location.

<br>

###  Signal Model
For a narrowband signal received by an $M$-element antenna array, the received signal vector is modelled as:

$\mathbf{x}(t) = \mathbf{a}(\theta)\, s(t) + \mathbf{n}(t)$

Where:

- $\mathbf{x}(t) \in  \mathbb{C}^M$ is the received signal vector
- $s(t)$ is the transmitted signal
- $\mathbf{n}(t)$ is additive noise
- $\mathbf{a}(\theta)$ is the **array steering vector**
- $\theta$ is the direction (AOA / DOA) of the incoming signal
> ***Reference:*** [link](https://ocw.mit.edu/courses/6-451-principles-of-digital-communication-ii-spring-2005/7cb3929341f072786598cd05c69a3f5c_chap_2.pdf#:~:text=2.1%20Continuous-time%20AWGN%20channel%20model.%20The%20continuous-time,density%20N0%20which%20is%20independent%20of%20X%28t%29.)

<br>

###  Steering Vector (Uniform Linear Array)
For a ULA with element spacing $d$ and signal wavelength $\lambda$, the [steering vector](https://www.sciencedirect.com/science/chapter/monograph/pii/B9780123743534000089) is defined as:

$$\mathbf{a}(\theta) = \begin{bmatrix}1 \\e^{-j \frac{2\pi d}{\lambda} \sin\theta} \\e^{-j \frac{4\pi d}{\lambda} \sin\theta} \\\vdots \\e^{-j \frac{2\pi (M-1) d}{\lambda} \sin\theta}\end{bmatrix}$$

To avoid spatial aliasing, the antenna spacing is constrained by:

$d \leq  \frac{\lambda}{2}$

At $5\ \text{GHz}$, this corresponds to a [maximum spacing](https://www.researchgate.net/publication/329501527_On_fundamental_operating_principles_and_range-doppler_estimation_in_monolithic_frequency-modulated_continuous-wave_radar_sensors) of approximately $3\ \text{cm}$.

<br>

###  DOA Estimation Using MUSIC
The **Multiple Signal Classification (MUSIC)** algorithm estimates DOA by exploiting the orthogonality between the signal and noise subspaces of the received signal covariance matrix.

The sample covariance matrix is computed as:

$$\hat{\mathbf{R}}_{xx} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}_n \mathbf{x}_n^H$$

Eigen-decomposition separates the noise subspace $\mathbf{E}_n$, and the MUSIC pseudospectrum is defined as:

$$P_{\text{MUSIC}}(\theta) =\frac{1}{\mathbf{a}^H(\theta)\, \mathbf{E}_n \mathbf{E}_n^H\, \mathbf{a}(\theta)}$$

Peaks in $P_{\text{MUSIC}}(\theta)$ correspond to estimated DOA angles.

<br>

###  Position Estimation Parameters

| Parameter | Symbol / Formula | Description |
|---------|------------------|-------------|
| Anchor position | $(x_i, y_i)$ | Cartesian coordinates of the anchor device equipped with an antenna array |
| Target position (ground truth) | $(x_T, y_T)$ | True Cartesian coordinates of the transmitting target device |
| Number of antenna elements | $M$ | Number of elements in the uniform linear array |
| Element spacing | $d$ | Distance between adjacent antenna elements (typically $\lambda/2$) |
| Carrier frequency | $f$ | Signal carrier frequency (fixed at $5\ \text{GHz}$) |
| Wavelength | $\lambda = \frac{c}{f}$ | Signal wavelength |
| Propagation speed | $c = 3 \times 10^8\ \text{m/s}$ | Speed of electromagnetic wave propagation |
| Snapshot count | $N$ | Number of temporal samples used to estimate covariance |
| Signal-to-noise ratio | $\text{SNR}$ | Controls noise level in the received signal |
| Number of sources | $K = 1$ | Number of dominant signal sources (single target assumption) |
| True DOA | $\theta$ | Angle between anchor array broadside and target direction |
| DOA noise | $\epsilon_\theta$ | Angular noise introduced by noise, interference, or NLOS |
| Observed AOA | $\theta^{\text{obs}} = \theta + \epsilon_\theta$ | Noisy AOA used for positioning and ML training |

<br>
<br>

###  Conceptual Example
####  Roles

-  **T** → Target device (transmitter, unknown position)
-  **A, B, C, …** → Anchors with antenna arrays (receivers, known positions)

---

###  Calculations
1.  **T transmits** a narrowband signal
2.  **Each anchor array** receives the signal across multiple antenna elements
3. Phase differences between elements encode the arrival angle
4. MUSIC estimates the dominant DOA $\theta$
5. Each DOA forms a **ray** originating from the anchor:

$$\mathbf{p}(t) = \mathbf{a}_i + t \begin{bmatrix} \cos\theta_i \\ \sin\theta_i  \end{bmatrix}$$

8. Target position is estimated by intersecting rays from multiple anchors or fusing with RSSI / TOA estimates

<br>

###  Visual Example

```javaScript
const network = {
  environment: {
    carrier_frequency: 5e9,
    propagation_speed: 3e8,
    doa_noise_std_dev: 3.0       // degrees
  },

  anchors: {
    A: {
      position: [-6.0, 4.5],
      antenna_array: {
        type: "ULA",
        num_elements: 8,
        element_spacing: 0.03    // λ/2 at 5 GHz
      }
    },

    B: {
      position: [8.0, 1.0],
      antenna_array: {
        type: "ULA",
        num_elements: 8,
        element_spacing: 0.03
      }
    }
  },

  target_device: {
    id: "T",
    position: [3.5, -2.0]
  },

  doa_measurements: {
    A: {
      true_angle: 0.82,
      observed_angle: 0.88,
      snapshot_count: 200,
      snr: 15
    },

    B: {
      true_angle: -1.34,
      observed_angle: -1.29,
      snapshot_count: 200,
      snr: 12
    }
  }
};
```
