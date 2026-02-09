#  Data generation 

## Overall specifications & constraints

The main purpose of this document is to define the specification and constraints for the data generation phase of a ***Machine Learning Device Positioning Prediction Model.***

We are emulating ***2D indoor and outdoor environments*** with multiple devices operating within a ***5GHz frequency band***. To avoid bias, we will ensure that in the data generation phase both scenarios have an equal chance of being generated.
  
**Simplified data generation process:**

1. Define the environment type (*indoor or outdoor*) based on a random selection process.

2. Generate network characteristics given the environment type.

3. Return the generated data in a structured format for positioning estimation calculations.


##  Grid

-  **Dimensionality:** The environment is modelled as a two-dimensional Cartesian coordinate space, with positions represented as $\text{(x,y)}$ coordinate pairs.

-  **Restricted Domain and Range (x,y):** $[-10 < x < 10, -10 < y < 10]$ representing a indoor or outdoor environment (20m x 20m = 400 m²), such as a shopping mall, field, or warehouse.

-  **Limited device generation:** We will constrain the amount of devices in the network to a manageable number (e.g., 15-30) to ensure the dataset is not too large for training while still providing sufficient coverage for accurate positioning.



##  RSSI

  

###  Path Loss Exponent (Environment-Specific)

  

The path loss exponent \( n \) characterises the average rate at which signal power decays with distance and depends on the propagation environment.

In the data generation process, \( n \) is sampled once per scenario and remains fixed for all links within that scenario to preserve spatial consistency.

  

*Assumed values used in this project are:*

  

| Environment Type | Description | Path Loss Exponent \( n \) |

|------------------|------------|----------------------------|

| Free Space / Open Outdoor | Minimal obstruction, near line-of-sight | 2.0 – 2.2 |

| Outdoor (Urban / Light Clutter) | Buildings, foliage | 2.7 – 3.5 |

| Indoor LOS | Open indoor areas (e.g. halls, warehouses) | 2.0 – 3.0 |

| Indoor NLOS | Multiple walls, partitions | 3.0 – 6.0 |

| Heavily Obstructed Indoor | Dense materials (concrete, metal) | 4.0 – 6.0 |

  

>***Reference:***  *Table 4.2 [Srinivasan, S.; Haenggi, M.,](https://arxiv.org/pdf/0802.0351v1.pdf) [Path loss exponent estimation in large wireless networks](https://arxiv.org/pdf/0802.0351v1.pdf)[, Information Theory and Applications Workshop, pp. 139, Feb 2009.](https://arxiv.org/pdf/0802.0351v1.pdf)*

  

<br>

  

###  Log-Distance Path Loss Model

  

The large-scale average path loss between two devices separated by distance \( d \) is modelled using the [log-distance path loss model](https://en.wikipedia.org/wiki/Log-distance_path_loss_model?utm_source=chatgpt.com#cite_note-Ref_1-3):

  

$PL(d) = PL(d_0) + 10 n \log_{10}\!\left(\frac{d}{d_0}\right)$

  

where:

  

- $PL(d)$ is the path loss at distance $d$ (dB)

- $d_0$ is a reference distance (typically $1\,\text{m}$)

- $n$ is the path loss exponent

- $PL(d_0)$ is the free-space path loss at $d_0$)

  

The free-space path loss at the reference distance is given by:

  

$PL(d_0) = 20  \log_{10}\!\left(\frac{4\pi d_0}{\lambda}\right)$

  

with wavelength $(\lambda = \frac{c}{f}$), where \( c \) is the speed of light and \( f \) is the carrier frequency (5 GHz).

  

<br>

  

###  Log-Normal Shadowing (Gaussian Noise)

  

To model environmental variability and measurement uncertainty, log-normal shadowing is applied by adding a **zero-mean Gaussian random variable** $(X_\sigma)$ in the logarithmic domain:

  

$PL(d) = PL(d_0) + 10 n \log_{10}\!\left(\frac{d}{d_0}\right) + X_\sigma$

  

where:

  

$X_\sigma  \sim  \mathcal{N}(0, \sigma^2)$

  

The parameter $( \sigma )$ represents the standard deviation of shadow fading (in dB) and is environment-dependent.

  

Typical values to potentially used in this project:

  

| Environment | $( \sigma )$ (dB) |

|------------|------------------|

| Free space / open outdoor | 1 – 2 |

| Indoor LOS | 2 – 4 |

| Indoor NLOS | 4 – 8 |

| Heavy obstruction | 6 – 10 |

  

The Gaussian noise term is applied independently to each wireless link, while $(\sigma)$ remains fixed per scenario.

  

> **NOTE:** If we want to be more precise, we can estimate an exact value of $( \sigma )$ from measured (or emulated) received-power samples by fitting the log-distance model using **minimum mean square error (MMSE)** and then computing the **sum of squared errors (SSE)** as shown below.

  

---

  

####  Estimating $(n)$ and $( \sigma )$ from Received-Power Samples (MMSE + SSE)

  

Assume we have $(k)$ received power measurements ${p_i}_{i=1}^{k}$ (in dBm) collected at distances ${d_i}_{i=1}^{k}$ from a transmitter, and we select a reference distance $d_0$ with known reference power $p(d_0)$.

  

####  1) Log-distance received-power model (in dBm)

  

The predicted received power at distance \(d_i\) is:

  

$\hat{p}_i = p_{i}(d_0) - 10n\log_{10}\!\left(\frac{d_i}{d_0}\right)$

  
  

####  2) Sum of Squared Errors (SSE)

  

Define the SSE cost function:

  

$J(n)=\sum_{i=1}^{k}\left(p_i-\hat{p}_i\right)^2$

  

where:

- $p_i$ is the measured received power at distance $d_i$

- $\hat{p}_i$ is the model-predicted received power at distance $d_i$

- Using the $(\frac{d}{d_{0}})^{n}$ path loss model

- From: $PL(d) = PL(d_0) + 10 n \log_{10}\!\left(\frac{d}{d_0}\right) + X_\sigma$

  

---

  

####  3) MMSE estimate of the path loss exponent $( \hat{n} )$

  

The MMSE estimate is the value of $n$ that minimises the squared error:

  

$\hat{n}=\arg\min_{n}J(n)$

  

To compute this estimate, differentiate $J(n)$ with respect to $n$ and set the derivative to zero:

  

$\frac{dJ(n)}{dn}=0$

  

Solving this equation yields $(\hat{n})$.

  
  

####  Estimating $( \sigma )$ using the SSE (as in the worked example)

  

Once $(\hat{n})$ is obtained, we can compute the SSE at the optimum:

  

$J(\hat{n})=\sum_{i=1}^{k}\left(p_i-\hat{p}_i(\hat{n})\right)^2$

  

Then estimate the (sample) shadowing variance using the same approach as the example at *[Srinivasan, S.; Haenggi, M.,](https://arxiv.org/pdf/0802.0351v1.pdf) [Path loss exponent estimation in large wireless networks](https://arxiv.org/pdf/0802.0351v1.pdf)[, Information Theory and Applications Workshop, pp. 144, Feb 2009.](https://arxiv.org/pdf/0802.0351v1.pdf)*

  

$\sigma^2=\frac{J(\hat{n})}{k}$

  

and the corresponding shadowing standard deviation:

  

$\sigma=\sqrt{\sigma^2}=\sqrt{\frac{J(\hat{n})}{k}}$

  

Finally, this $(\sigma)$ is then used as the scenario-level log-normal shadowing parameter:

  

$X_\sigma  \sim  \mathcal{N}(0,\sigma^2)$

  

<br>

  

###  Using $( \sigma )$ during data generation

  

For each scenario:

  

1. Fit $(\hat{n})$ using MMSE (minimise $J(n)$).

2. Compute $(\sigma = \sqrt{J(\hat{n})/k})$.

3. For each wireless link $(i \rightarrow j)$, sample an independent shadowing term:

  

$X_{\sigma,i,j}=\sigma z,\;\;\; z\sim  \mathcal{N}(0,1)$

  

| Quantity | Role |

|----------|------|

| $(\sigma)$ | Controls the *spread* (standard deviation) of the shadowing noise |

| $(z)$ | Standard normal random variable with unit variance, $z \sim \mathcal{N}(0,1)$ |

| $(X_\sigma = \sigma z)$ | Actual log-normal shadowing noise value added to the path loss (in dB) |

  
  

4. Apply it to emulate received power (equivalently, apply it as an additive term in the dB-domain model):

  

$p_{r}(d)=p(d_0)-10\hat{n}\log_{10}\!\left(\frac{d}{d_0}\right)+X_{\sigma,i,j}$

  

This preserves the intended design:

- $(\sigma)$ is **fixed per scenario**

- shadowing noise $X_{\sigma,i,j}$ is **independent per link**

  

<br>

  

###  Calculating Received Signal Strength Indicators (RSSI)

  

Once the total path loss is computed (including free-space loss, material attenuation, reflection loss, and shadowing), the received signal strength is calculated using:

  

$RSSI_{i,j} = P_t + G_{Tx} + G_{Rx} - PL_{i,j}$

  

where:

  

- $P_t$ is the transmit power (dBm)

- $G_{Tx}$ and $G_{Rx}$ are the transmitter and receiver antenna gains (dBi)

- $PL_{i,j}$ is the total path loss along the wireless channel between devices $i \rightarrow j$

  

<br>

  

###  RSSI - Devices and Channels Representation

  

The generated dataset is organised into device-level and channel-level components.

Ground-truth geometry is used to compute distances and deterministic path loss, while RSSI values are treated as noisy observations influenced by environment-specific parameters and log-normal shadowing.

  

<br>

  

| Devices | Type | Description |

|-----|------|-------------|

| `devices` | Dictionary | Container for all devices participating in the network scenario. |

| `devices.targetDevice` | Object | Target device whose position is to be estimated by classical algorithms and the ML model. |

| `devices.targetDevice.position` | Array `[x, y]` | Ground-truth Cartesian coordinates of the target device (meters). |

| `devices.otherDevices` | Dictionary | Collection of reference (anchor) devices with known positions. |

| `devices.otherDevices.{id}.position` | Array `[x, y]` | Cartesian coordinates of reference device `{id}` (meters). |

  

<br>

<br>

<br>

  

| Environment Parameters | Type | Description |

|-----|------|-------------|

| `environment.type` | String | Environment category (`"indoor"` or `"outdoor"`). |

| `environment.losModel` | String | Propagation condition model (`"LOS"` or `"NLOS"`). |

| `environment.pathLossExponent` | Float | Scenario-level path loss exponent $(n)$ sampled once per scenario. |

| `environment.shadowingStdDev` | Float | Scenario-level shadowing standard deviation $(\sigma)$ in dB. |

  

<br>

<br>

<br>

  

| Channels | Type | Description / Formula |

|-----|------|------------------------|

| `channels` | Dictionary | Collection of wireless links between device pairs, indexed by identifiers (e.g. `"T-A"`). |

| `channels.{i-j}.distance` | Float | Euclidean distance between devices $i$ and $j$ (meters): $d_{i,j} = \sqrt{(x_i-x_j)^2 + (y_i-y_j)^2}$ |

| `channels.{i-j}.freeSpacePathLoss` | Float | Free-space path loss at distance $d_{i,j}$: $PL(d_0)+10n\log_{10}\!\left(\frac{d_{i,j}}{d_0}\right)$ |

| `channels.{i-j}.shadowing` | Float | Log-normal shadowing term sampled as $X_{\sigma,i,j}\sim\mathcal{N}(0,\sigma^2)$ (dB). |

| `channels.{i-j}.pathLoss` | Float | Total path loss (dB): sum of deterministic loss terms and shadowing. |

| `channels.{i-j}.receivedSignalStrength` | Float | RSSI at receiver (dBm): $RSSI_{i,j}=P_t+G_{Tx}+G_{Rx}-PL_{i,j}$ |

  

<br>

<br>

  

---

  

####  Visual Example :

#####  Aligned with Log-Distance + Shadowing Model

  

```javascript

const network = {

environment: {

type: "indoor",

losModel: "NLOS",

pathLossExponent: 4.2, // n (fixed per scenario)

shadowingStdDev: 6.0  // σ in dB (fixed per scenario)

},

  

devices: {

targetDevice: {

id: "T",

position: [3.5, -2.0]

},

  

otherDevices: {

A: { position: [-6.0, 4.5], isReference: false },

B: { position: [8.0, 1.0], isReference: false },

C: { position: [-2.0, -7.5], isReference: false },

D: { position: [5.0, -6.0], isReference: false }

}

},

  

channels: {

"T-A": {

distance: 11.2,

los: false,

freeSpacePathLoss: 65.8,

shadowing: 2.3,

logDistancePathLoss: 75.5,

receivedSignalStrength: -62.3

},

  

"T-B": {

distance: 6.8,

los: true,

obstaclesCrossed: [],

freeSpacePathLoss: 61.4,

shadowing: -1.2,

logDistancePathLoss: 60.2,

receivedSignalStrength: -55.0

}

}

};

```