# Explicit ASIF

For explicit ASIF RTA, each barrier constraint can be constructed explicitly as,

$$
    BC_i(\boldsymbol{x},\boldsymbol{u}) := \nabla h_i(\boldsymbol{x}) (f(\boldsymbol{x}) + g(\boldsymbol{x})\boldsymbol{u}) + \alpha(h_i(\boldsymbol{x})),
$$

where $\boldsymbol{u} \in U$ and $h_{i}(\boldsymbol{x})$ refers to the set of $M$ control invariant safety constraints.

The ASIF filter then solves the quadratic program using the set of $i$ barrier constraints, and outputs $\boldsymbol{u}_{\rm safe}(\boldsymbol{x})$ according to the following logic,

$$
\begin{split}
\boldsymbol{u}_{\text{safe}}(\boldsymbol{x})=\underset{\boldsymbol{u} \in \mathcal{U}}{\text{argmin}} & \Vert \boldsymbol{u}_{\text{des}}-\boldsymbol{u}\Vert ^{2} \\
\text{s.t.} \quad & BC_i(\boldsymbol{x},\boldsymbol{u})\geq 0, \quad \forall i \in \{1,...,M\}
\end{split}
$$

It is important to note that when the relative degree of the system is greater than one, i.e. when $\dot{h}_i(\boldsymbol{x})$ does not depend on $\boldsymbol{u}$, the barrier constraint defined above will no longer be able to enforce safety because $\boldsymbol{u}$ vanishes from the equation. Therefore, a new constraint $\Psi_i(\boldsymbol{x})$ must be defined such that $\dot{\Psi}_i(\boldsymbol{x})$ depends on $\boldsymbol{u}$. For the case where $\nabla h_i (\boldsymbol{x}) g(\boldsymbol{x}) \equiv 0$, $\Psi_i(\boldsymbol{x})$ is defined as,

$$
    \Psi_i(\boldsymbol{x})=\nabla h_i (\boldsymbol{x})f (\boldsymbol{x})+ \alpha(h_i(\boldsymbol{x})).
$$

The barrier constraint can then be defined as,

$$
    BC_i(\boldsymbol{x},\boldsymbol{u}) := \nabla \Psi_i(\boldsymbol{x}) (f(\boldsymbol{x}) + g(\boldsymbol{x})\boldsymbol{u}) + \alpha(\Psi_i(\boldsymbol{x})).
$$

If the relative degree of the system is two, then $\dot{\Psi}_i(\boldsymbol{x})$ as defined above will depend on $\boldsymbol{u}$ and the barrier constraint is valid. If the relative degree of the system is greater than two, then this process can be repeated, where,

$$
    \Psi_{2_i}(\boldsymbol{x})=\nabla \Psi_{1_i} (\boldsymbol{x})f (\boldsymbol{x})+ \alpha(\Psi_{1_i}(\boldsymbol{x})).
$$
