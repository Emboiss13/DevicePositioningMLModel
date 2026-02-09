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