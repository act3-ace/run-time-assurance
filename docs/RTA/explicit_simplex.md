# Explicit Simplex

The Simplex filter can be designed using the explicit definition of $\mathcal{C}_{\rm S}$,

$$
    \mathcal{C}_{\rm S} := \{\boldsymbol{x} \in \mathcal{X} \, | \, h_i(\boldsymbol{x}) \geq 0, \forall i \in \{1,...,M\} \}.
$$

The simplex filter then determines if $\phi_1^{\boldsymbol{u}_{\rm des}}(\boldsymbol{x}) \in \mathcal{C}_{\rm S}$, and outputs $\boldsymbol{u}_{\rm safe}(\boldsymbol{x})$ according to the following logic,

$$
\begin{array}{rl}
\boldsymbol{u}_{\rm safe}(\boldsymbol{x})=
\begin{cases}
\boldsymbol{u}_{\rm des}(\boldsymbol{x}) & {\rm if}\quad \phi_1^{\boldsymbol{u}_{\rm des}}(\boldsymbol{x}) \in \mathcal{C}_{\rm S}  \\
\boldsymbol{u}_{\rm b}(\boldsymbol{x})  & {\rm if}\quad otherwise
\end{cases}
\end{array}
$$

For an explicit Simplex RTA, the backup controller should be designed such that $\boldsymbol{u}_{\rm b}$ will always keep the system in $\mathcal{C}_{\rm S}$.
