# Defining Safety

To formally define safety, a set of $M$ inequality constraints $\varphi_i(\boldsymbol{x}): \mathcal{X}\to \mathbb{R}$,  $\forall i \in \{1,...,M\}$ can be developed, where $\varphi_i(\boldsymbol{x}) \geq 0$ when the constraint is satisfied. The set of states that satisfies this collection of inequality constraints is referred to as the *allowable set*, and is defined as,
$$
    \mathcal{C}_{\rm A} := \{\boldsymbol{x} \in \mathcal{X} \, | \, \varphi_i(\boldsymbol{x}) \geq 0, \forall i \in \{1,...,M\} \}.
$$

For a dynamical system with no control input, an intial state $\boldsymbol{x}(t_0)$ is *safe* if it remains within $\mathcal{C}_{\rm A}$ for all time. This is know as *forward invariance*. The *safe set*, a forward invariant subset of $\mathcal{C}_{\rm A}$, is defined as,

$$
    \boldsymbol{x}(t_0) \in \mathcal{C}_{\rm S} \Longrightarrow \boldsymbol{x}(t) \in \mathcal{C}_{\rm A}, \forall t \geq t_0.
$$

Often in control systems, there are states that lie in $\mathcal{C}_{\rm A}$ at the current time but will not remain in $\mathcal{C}_{\rm A}$ for all time, typically due to limitations on control as defined by the admissible control set $\mathcal{U}$. Therefore, $\mathcal{C}_{\rm S}$ must also be a *control invariant* subset of $\mathcal{C}_{\rm A}$, where there exists a control law $\boldsymbol{u}\in \mathcal{U}$ that renders $\mathcal{C}_{\rm S}$ forward invariant.

## Explicit Safe Set

$\mathcal{C}_{\rm S}$ can be defined *explicitly*, where a set of $M$ control invariant inequality constraints $h_i(\boldsymbol{x}): \mathcal{X}\to \mathbb{R}$, $\forall i \in \{1,...,M\}$ are developed, where $h_i(\boldsymbol{x}) \geq 0$ when the constraint is satisfied. There must exist a control law $\boldsymbol{u}\in \mathcal{U}$ that renders all of the constraints forward invariant. $\mathcal{C}_{\rm S}$ is then defined explicitly as,

$$
    \mathcal{C}_{\rm S} := \{\boldsymbol{x} \in \mathcal{X} \, | \, h_i(\boldsymbol{x}) \geq 0, \forall i \in \{1,...,M\} \}.
$$

The explicit safe set can be determined entirely offline, however this often becomes difficult for higher dimensional and more complex systems. The system can verify that $\boldsymbol{x}\in \mathcal{C}_{\rm S}$ by simply verifying that $h_i(\boldsymbol{x})\geq0, \forall i \in \{1,...,M\}$.

## Implicit Safe Set

$\mathcal{C}_{\rm S}$ can also be defined *implicitly* using closed loop trajectories under a predetermined backup control law $\boldsymbol{u}_{\rm b}$. This backup control law will guide the system to a backup set $\mathcal{C}_{\rm B} \subseteq \mathcal{C}_{\rm S}$, where the system is verified to remain safe for all time. $\mathcal{C}_{\rm S}$ is then defined implicitly as,

$$
    \mathcal{C}_{\rm S} := \{\boldsymbol{x}\in \mathcal{X} \, | \, \forall t\geq 0,\,\, \phi^{\boldsymbol{u}_{\rm b}}(t;\boldsymbol{x})\in\mathcal{C}_{\rm A} \},
$$

where $\phi^{\boldsymbol{u}_{\rm b}}$ represents a prediction of the state $\boldsymbol{x}$ for $t$ seconds under the backup control law $\boldsymbol{u}_{\rm b}$. The implicit safe set is determined entirely online, where backup trajectories must constantly be computed at each time interval.

In practice, these trajectories are often evaluated over a finite time horizon $\mathcal{T} = [0,T]$ in order to reduce computation time. As a result, a trade-off between computation time and safety guarantees is introduced: as the length of the time horizon decreases, the computation time decreases but safety is guaranteed over a shorter time horizon. As the length of the time horizon increases, the computation time increases while safety is guaranteed over a longer time horizon.
