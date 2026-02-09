
# Data generation specifications & constraints

The main purpose of this document is to define the specification and constraints for the data generation phase of a 5GHz Machine Learning Device Positioning Prediction Model. 

We are emulating 2D indoor and outdoor environment with multiple devices operating within a 5GHz frequency band. To avoid bias, we will ensure that in the data generation phase both scenarios have an equal chance of being generated.

**Simplified data generation process:**
1. Define the environment type (*indoor or outdoor*) based on a random selection process.
2. Generate network characteristics given the environment type.
3. Return the generated data in a structured format (e.g., a dictionary) for positioning estimation calculations.


## Grid
- **Dimensionality:** The environment is modelled as a two-dimensional Cartesian coordinate space, discretised into uniform spatial divisions, with positions represented as $`\text{(x,y)}`$ coordinate pairs.

- **Restricted Domain and Range (x,y):** $`[-10 < x < 10, -10 < y < 10]`$ representing a typical large-scale indoor or campus-like environment (20m x 20m = 400 m²), such as a shopping mall or warehouse.

- **Limited device generation:** We will constrain the amount of devices in the network to a manageable number (e.g., 15-30) to ensure the dataset is not too large for training while still providing sufficient coverage for accurate positioning.


## RSSI

Path-loss and RSSI values are derived from ground-truth distances during data generation, while RSSI-based distance estimates are treated as noisy observations during positioning.

For non-line-of-sight (NLOS) and complex outdoor propagation scenarios, the path-loss model extends beyond free-space assumptions by incorporating additional attenuation due to obstacles and material interactions. 

- In these cases, the total path loss is modelled as the sum of the free-space path loss and environment-specific excess losses. 
- These excess losses account for signal attenuation caused by transmission through obstructing materials (e.g., walls or partitions), modelled using material-dependent attenuation coefficients and obstacle thickness, as well as additional refraction or diffraction losses introduced when the signal propagates around edges or corners. 

### Material Attenuation Coefficients


### Devices and Channels Representation

| Parameter | Type | Description / Formula |
|----------|------|------------------------|
| `devices` | Dictionary | Container for all devices participating in the network scenario. |
| `targetDevice` | Object | The device whose position is to be estimated by the positioning algorithms and ML model. |
| `targetDevice.position` | Array `[x, y]` | Ground-truth Cartesian coordinates of the target device in meters. |
| `otherDevices` | Dictionary | Collection of reference devices (anchors) with known positions, indexed by unique identifiers. |
| `otherDevices.{id}.position` | Array `[x, y]` | Cartesian coordinates of a reference device in meters. |
| `channels` | Dictionary | Collection of wireless links between pairs of devices, indexed by device pair identifiers (e.g., `"T-A"`). |
| `channels.{i-j}.distance` | Float | Euclidean distance between devices *i* and *j* (in meters): $`\displaystyle d_{i,j} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}`$ |
| `channels.{i-j}.pathLoss` | Float | Propagation loss along the wireless channel (in dB), computed using either a free-space or environment-specific path-loss model. For free space:  $`\text{FSPL\ (dB)}=20\log _{10}(d)+20\log _{10}(f)-\text{GTx}-\text{GRx}`$ |
| `channels.{i-j}.receivedSignalStrength` | Float | Received Signal Strength Indicator (RSSI) at the receiver (in dBm), derived from the Friis transmission equation:  $`\displaystyle P_r = P_t + G_{Tx} + G_{Rx} - \text{PathLoss}_{i,j}`$ |
| `obstacles` | Array of Objects | List of obstacles present in the environment. Each obstacle object typically includes geometry (e.g., polygon/rectangle), material type, thickness, and (optionally) surface electrical properties used to model attenuation and reflections. |
| `obstacles[k].material` | String | Material label (e.g., `"concrete"`, `"glass"`, `"drywall"`) used to look up attenuation/reflection parameters. |
| `obstacles[k].thickness` | Float | Obstacle thickness \(t_k\) in meters. Used for through-material attenuation. |
| `obstacles[k].attenuationCoeff` | Float | Material attenuation coefficient $`\alpha_{m_k}`$ in dB/m. |
| `obstacles[k].materialAttenuation` | Float | Signal loss due to transmission through the obstacle (in dB):  $` L_{\text{mat},k} = \alpha_{m_k}\, t_k `$ |
| `obstacles[k].reflectionLoss` | Float | Additional loss (in dB) due to reflection at the material boundary |


### Visual Example
```javaScript
const network = {
  // Device positions in 2D Cartesian space
  devices: {
    targetDevice: {
      id: "T",
      position: [3.5, -2.0] // [x, y] in meters
    },

    otherDevices: {
      A: { position: [-6.0, 4.5] },
      B: { position: [8.0, 1.0] },
      C: { position: [-2.0, -7.5] },
      D: { position: [5.0, -6.0] }
    }
  },

  // Environmental obstacles affecting propagation
  obstacles: [
    {
      id: "W1",
      material: "concrete",
      thickness: 0.25,              // meters
      attenuationCoeff: 12.0,        // dB/m
      reflectionCoeff: 0.6           // |Γ|
    },
    {
      id: "W2",
      material: "glass",
      thickness: 0.02,
      attenuationCoeff: 3.5,
      reflectionCoeff: 0.2
    }
  ],

  // Channel-level properties per device pair
  channels: {
    "T-A": {
      distance: 11.2,                // meters
      los: false,
      obstaclesCrossed: ["W1"],
      freeSpacePathLoss: 66.1,        // dB
      materialLoss: 3.0,              // dB (α · t)
      reflectionLoss: 4.4,             // dB (-20 log10 |Γ|)
      pathLoss: 73.5,                 // dB (total)
      receivedSignalStrength: -62.3   // dBm
    },

    "T-B": {
      distance: 6.8,
      los: true,
      obstaclesCrossed: [],
      freeSpacePathLoss: 61.4,
      materialLoss: 0.0,
      reflectionLoss: 0.0,
      pathLoss: 61.4,
      receivedSignalStrength: -55.0
    },

    "T-C": {
      distance: 5.9,
      los: false,
      obstaclesCrossed: ["W2"],
      freeSpacePathLoss: 60.2,
      materialLoss: 0.07,
      reflectionLoss: 1.9,
      pathLoss: 62.2,
      receivedSignalStrength: -58.2
    },

    "T-D": {
      distance: 7.4,
      los: false,
      obstaclesCrossed: ["W1", "W2"],
      freeSpacePathLoss: 62.0,
      materialLoss: 3.07,
      reflectionLoss: 6.3,
      pathLoss: 71.4,
      receivedSignalStrength: -60.1
    }
  }
};


```



# Formulas

**Free Space Path Loss:**
$`\text{FSPL\ (dB)}=20\log _{10}(d)+20\log _{10}(f)-\text{GTx}-\text{GRx}`$

-   $`d`$ = distance (in meters)
-   $`f`$ = frequency (in Hz)
-   $`c`$ = speed of light (3×108  m/s)
-   $`GTx`$ = transmitter antenna gain (in dBi)
-   $`GRx`$ = receiver antenna gain (in dBi)

**Friis Equation:**
$`P_{r} = \left( P_{t} \right) \left( G_{t} \right) \left( G_{r} \right) \left( \frac{λ}{(4πd)^2}\right)`$

-   $`d`$ = distance (in meters)
-   $`f`$ = frequency (in Hz)
-   $`c`$ = speed of light (3×108  m/s)
-   $`GTx`$ = transmitter antenna gain (in dBi)
-   $`GRx`$ = receiver antenna gain (in dBi)
