## 5GHz Machine Learning Device Positioning Prediction Model

### Project Description
The Global Positioning System (GPS) is widely used for outdoor positioning and
navigation. However, GPS relies on satellite signals that are significantly attenuated
by buildings and obstacles, making it unreliable indoors. As a result, alternative
approaches are required to enable accurate device positioning in scenarios with multipath interference and signal attenuation [4].

This project aims to address the limitations of GPS by developing a machine learning
(ML)-based positioning prediction model that operates independently of satellite
signals and instead leverages communication between devices within the same 5 GHz
localised network. To achieve this, the project will evaluate and compare signal-based,
direction-based and time-based indoor positioning methods [2], under emulated
conditions. A key assumption in this project is that multi-path interference will not
significantly affect localisation performance. Therefore, while multi-path propagation
will be considered, the system will focus only on the strongest received signal when
estimating position. Moreover, the specific problem addressed in this project is
whether a machine learning model can outperform conventional indoor positioning
methods by learning patterns and compensating for environmental conditions that
arise in both indoor and outdoor propagation environments, such as noise,
interference, non-line-of-sight (NLOS), and line-of-sight (LOS) [1].

This problem provides a sufficient challenge for an undergraduate dissertation due to the mathematical
complexity of using device positioning methods and the intricacies of generating
realistic network behaviours. Furthermore, the project requires evaluating multiple
positioning methods, as well as designing and training a machine learning model.
Hence, this project will demonstrate appropriate technical depth, independent
problem-solving, and critical analysis for a computer science dissertation.

This project is sponsored by Clear-Com, a telecommunications company that provides
hardware and embedded software solutions for a range of professional sectors,
including entertainment, nuclear facilities, military operations, and space exploration. 
Since Clear-Com devices are used in both indoor and outdoor environments, where
accurate device positioning can be crucial, they can particularly benefit from this
project. However, many organisations face the same issues and constraints, allowing
for the outcomes of this project to be applicable beyond a single company or use case.

<br>

## Aims & Objectives
The overall aim of this project is to investigate whether a machine learning model can
outperform conventional indoor positioning methods by learning patterns and
compensating for variations caused by network constraints and environmental
conditions.

The project will be structured into four main stages, each targeting a specific objective
which forms a sequential pipeline in which the output of each stage is used as the
input to the next.

1. **Generate network specifications & constraints:** Multiple two-dimensional
network environments will be emulated by randomly generating scenario
parameters and constraints. Each environment scenario will include all data
required to perform positioning estimations. This includes generating the exact
(ground-truth) position of the target device and other devices in the network,
obstacles, material attenuation coefficients, noise, interference, signal strength
measurements, etc.

    - **Risk**: Failing to represent realistic network conditions or introducing bias
by lacking enough variation in the generated data.

    - **Mitigation**: Generating data that offers a representative snapshot of
diverse network reception conditions, including varying levels of noise,
interference, and propagation conditions, to ensure sufficient diversity
and relevance [1].

<br>

2. **Calculate position estimations:** The generated network environments will be
used to produce multiple position estimates. For each network scenario, the
position of the same target device will be calculated using three indoor
positioning methods: Received Signal Strength Indicators (RSSI), Time of
Arrival (TOA), and Angle of Arrival (AOA).

    - **Risk**: Emulated environments may oversimplify network conditions
affecting the performance of some positioning methods more than
others.
    - **Mitigation**: Attenuation, noise, and LOS/NLOS conditions will be
incorporated to improve dataset realism [1].

<br>

3. **Construct a labelled dataset and train the ML model:** The results from the
positioning estimation phase will be mapped back to their corresponding
network environments to form a labelled training dataset. This dataset will be
used to train the ML model to analyse patterns between the ground-truth device
position, the different estimated positions, and the related environmental
constraints for each network scenario.

    - **Risk**: The dataset size may be insufficient to support a reliable machine learning
learning performance, leading to high variance, overfitting, or inflated
accuracy estimates [3].
    - **Mitigation**: The dataset size will be incrementally increased until the model
performance stabilises, and regularisation techniques will be used to
prevent overfitting [3].

<br>

4. **Evaluate the ML model**: The trained ML model will be assessed to determine
whether it outperforms individual conventional indoor positioning methods.

    - **Risk**: The model may not achieve significant improvements.
    - **Mitigation**: If this occurs, the model may instead be repurposed to assist
conventional positioning methods by refining existing measurements or
correcting systematic errors, rather than directly predicting the device
positions [1].

<br>

## References
1. Alawieh, M. and Kontes, G. 5G positioning advancements with AI/ML. arXiv
preprint, arXiv:2401.02427, 2023. Available at:
https://arxiv.org/abs/2401.02427 [accessed March 2023].
2. Rathnayake, R.M.M.R., Maduranga, M.W.P., Tilwari, V. and Dissanayake,
M.B. RSSI and machine learning-based indoor localization systems for smart
cities. Eng, 4(2), pp. 1468–1494. Available at:
https://doi.org/10.3390/eng4020085 [accessed March 2023].
3. Rajput, D., Wang, W.J. and Chen, C.C. Evaluation of a decided sample size
in machine learning applications. BMC Bioinformatics, 24, p. 48, 2023.
Available at: https://doi.org/10.1186/s12859-023-05156-9 [accessed March
2023].
4. Xie, T., Jiang, H., Zhao, X. and Zhang, C. A Wi-Fi-based wireless indoor
position sensing system with multipath interference mitigation. Sensors,
19(18), p. 3983. Available at: https://doi.org/10.3390/s19183983 [accessed
March 2023].

---

#### Authors
Giuliana E, Victor Romero Cano, Juan Hernandez Vega & Arnold Chau
