# Discrete ASIF RTA Filters

For discrete time systems given by,

$$
    \boldsymbol{x}_{t + \Delta t} = f(\boldsymbol{x}_t, \boldsymbol{u}_t, \Delta t),
$$

safety of the system can be enforced through [Discrete Control Barrier Functions](https://hybrid-robotics.berkeley.edu/publications/RSS2017_Discrete_CBF.pdf), where the discrete time derivative of $h(\boldsymbol{x})$ is enforced as,

$$
        \Delta h(\boldsymbol{x}) = \frac{h(\boldsymbol{x}_{t + \Delta t}) - h(\boldsymbol{x}_t)}{\Delta t} \ge 0.
$$

Each discrete barrier constraint then becomes,

$$
    DBC_i(\boldsymbol{x},\boldsymbol{u}) := \Delta h_i(\boldsymbol{x}) + \alpha(h_i(\boldsymbol{x})) \ge 0.
$$

In this case, the barrier constraints often become nonlinear with respect to $\boldsymbol{u}$ due to the computation of $\boldsymbol{x}_{t + \Delta t}$. As a result, a quadratic program can no longer be used to solve for $\boldsymbol{u}_{\rm safe}$, and instead a nonlinear optimizer is used. Using the set of $i$ discrete barrier constraints, the ASIF filter outputs $\boldsymbol{u}_{\rm safe}(\boldsymbol{x})$ according to the following logic,

$$
\begin{split}
\boldsymbol{u}_{\text{safe}}(\boldsymbol{x})=\underset{\boldsymbol{u} \in \mathcal{U}}{\text{argmin}} & \Vert \boldsymbol{u}_{\text{des}}-\boldsymbol{u}\Vert ^{2} \\
\text{s.t.} \quad & DBC_i(\boldsymbol{x},\boldsymbol{u})\geq 0, \quad \forall i \in \{1,...,M\}
\end{split}
$$
