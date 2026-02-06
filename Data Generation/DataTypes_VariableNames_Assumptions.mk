# Data Types, Variable Names, and Assumptions for emulating Indoor/Outdoor (IO) network scenarios

The main purpose of this document is to define the specification and constraints for a 5GHz Machine Learning Device Positioning Prediction Model.
We are emulating an 2D indoor environment with multiple access points (APs) operating in the 5GHz frequency band.

This includes:
- Data types
- Variable names
- Assumptions 
- Formulas
- Constants

Since we want to train a machine learning model to accurately predict device positions (regardless of indoor/outdoor conditons) based on signal characteristics, we need to ensure that the data we generate is realistic and representative of real-world scenarios. 
This involves making certain assumptions about the environment, the behavior of signals, and the distribution of access points.
At the same time, we need to ensure that the training scenarios have both indoor and outdoor characteristics. 
To do this, we will ensure that in the data generation phase both scenarios have an equal chance of being generated.

Simplified data generation process:
1. Define the environment type (indoor or outdoor) based on a random selection process.
2. Generate network characteristics given the environment type.
3. Return the generated data in a structured format (e.g., a dictionary) for positioning estimation calculations.


## Indoor Positioning Environments and Network Scenario Data Generation

General asssumptions:

- Dimensionality: 2 dimensional (x,y) coordinates for both access points and target device positions.
- Restricted Domain and Range (x,y): [-50 < x < 50, -50 < y < 50] to represent a typical indoor environment of 100m^2 (e.g., office building, shopping mall).
- We will constrain the amount of APs to a manageable number (e.g., 10-15) to ensure the dataset is not too large for training while still providing sufficient coverage for accurate positioning.
- The access points are strategically placed to provide coverage throughout the indoor area.
- For indoor environments the LOS (Line of Sight) and NLOS (Non-Line of Sight) conditions will be considered, with a certain percentage of APs being in LOS and the rest in NLOS.

	https://gist.github.com/ifindev/5241f2a30defa7c683dd1faf36082c5a
	- Formula for Path loss in LOS = 
	- Formula for Path loss in NLOS =



| Data Type   | Variable Name | Assumption/Description |
| ----------- | ------------- | ---------------------- |
| dict        | accessPoints  | A dictionary containing the details of access points in a 5GHzindoor environment, including their locations and signal characteristics. |

