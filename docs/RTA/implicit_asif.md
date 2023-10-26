# Implicit ASIF

For implicit ASIF RTA, each barrier constraint can be constructed implicit as,

$$
    BC_i(\boldsymbol{x},\boldsymbol{u}) := \nabla \varphi_i(\phi^{\boldsymbol{u}_{\rm b}}_j) D(\phi^{\boldsymbol{u}_{\rm b}}_j) [f(\boldsymbol{x}) + g(\boldsymbol{x})\boldsymbol{u}] + \alpha(\varphi_i(\phi^{\boldsymbol{u}_{\rm b}}_j) ),
$$

where $\boldsymbol{u} \in U$, $\varphi_i(\boldsymbol{x})$ refers to the set of $M$ safety constraints that define $\mathcal{C}_{\rm A}$, $\phi^{\boldsymbol{u}_{\rm b}}_j$ represents a prediction of the state $\boldsymbol{x}$ at the $j^{th}$ discrete time interval along the backup trajectory $\forall t \in [0,\infty)$, and $D(\phi^{\boldsymbol{u}_{\rm b}}_j)$ is computed by integrating a sensitivity matrix along the backup trajectory. In practice, the trajectory is evaluated over a finite set of $j$ points over the interval $[0,T]$, where a barrier constraint is constructed at each point.

The ASIF filter then solves the quadratic program using the set of $i$ barrier constraints, and outputs $\boldsymbol{u}_{\rm safe}(\boldsymbol{x})$ according to the following logic,

$$
\begin{split}
\boldsymbol{u}_{\text{safe}}(\boldsymbol{x})=\underset{\boldsymbol{u} \in \mathcal{U}}{\text{argmin}} & \Vert \boldsymbol{u}_{\text{des}}-\boldsymbol{u}\Vert ^{2} \\
\text{s.t.} \quad & BC_i(\boldsymbol{x},\boldsymbol{u})\geq 0, \quad \forall i \in \{1,...,M\}
\end{split}
$$

Additionally, rather than constructing a barrier constraint at each point along the backup trajectory, these points can be subsampled such that the barrier constrains are only constructed at the point(s) where $\varphi_i(\phi^{\boldsymbol{u}_{\rm b}}_j)$ is minimized. This allows the ASIF to only consider the most relevant points along the trajectory.
