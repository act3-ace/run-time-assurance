# Implicit Simplex

The Simplex filter can be designed using the implicit definition of $\mathcal{C}_{\rm S}$,

$$
    \mathcal{C}_{\rm S} := \{\boldsymbol{x}\in \mathcal{X} \, | \, \forall t\geq 0,\,\, \phi^{\boldsymbol{u}_{\rm b}}(t;\boldsymbol{x})\in\mathcal{C}_{\rm A} \}.
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

For an implicit Simplex RTA, the backup controller should be designed using a control law that always leads the system back to $\mathcal{C}_{\rm B}$, which is verified to be safe.
