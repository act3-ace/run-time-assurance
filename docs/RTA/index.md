# Run Time Assurance Concepts

## Introduction

Run Time Assurance (RTA) is an online safety assurance technique that filters potentially unsafe inputs from a primary controller in a way that preserves safety of the system when necessary. RTA is designed to be completely independent of the primary control structure, such that it can be used with any control system. 

## Modeling Dynamical Control Systems

This module supports dynamical control systems, where $\boldsymbol{x} \in \mathcal{X} \subseteq \mathbb{R}^n$ denotes the state vector of size $n$ and $\boldsymbol{u}\in \mathcal{U} \subseteq\mathbb{R}^m$ denotes the control vector of size $m$. Here, $\mathcal{X}$ is a set defining all possible state values for the system, and $\mathcal{U}$ is a set of all admissible controls for the system, which is determined by the actuation constraints of the real world system. Assuming control affine dynamics, a continuous-time system model is given by a system of ordinary differential equations where,
$$
   \boldsymbol{\dot{x}} = f(\boldsymbol{x}) + g(\boldsymbol{x})\boldsymbol{u}.
$$

## Defining Safety

To formally define safety, a set of $M$ inequality constraints $\varphi_i(\boldsymbol{x}): \mathcal{X}\to \mathbb{R}$,  $\forall i \in \{1,...,M\}$ can be developed, where $\varphi_i(\boldsymbol{x}) \geq 0$ when the constraint is satisfied. The set of states that satisfies this collection of inequality constraints is referred to as the *allowable set*, and is defined as,
$$
    \mathcal{C}_{\rm A} := \{\boldsymbol{x} \in \mathcal{X} \, | \, \varphi_i(\boldsymbol{x}) \geq 0, \forall i \in \{1,...,M\} \}.
$$

For a dynamical system with no control input, an intial state $\boldsymbol{x}(t_0)$ is *safe* if it remains within $\mathcal{C}_{\rm A}$ for all time. This is know as *forward invariance*. The *safe set*, a forward invariant subset of $\mathcal{C}_{\rm A}$, is defined as, 

$$
    \boldsymbol{x}(t_0) \in \mathcal{C}_{\rm S} \Longrightarrow \boldsymbol{x}(t) \in \mathcal{C}_{\rm A}, \forall t \geq t_0.
$$

Often in control systems, there are states that lie in $\mathcal{C}_{\rm A}$ at the current time but will not remain in $\mathcal{C}_{\rm A}$ for all time, typically due to limitations on control as defined by the admissible control set $\mathcal{U}$. Therefore, $\mathcal{C}_{\rm S}$ must also be a *control invariant* subset of $\mathcal{C}_{\rm A}$, where there exists a control law $\boldsymbol{u}\in \mathcal{U}$ that renders $\mathcal{C}_{\rm S}$ forward invariant.

### Explicit Safe Set

$\mathcal{C}_{\rm S}$ can be defined *explicitly*, where a set of $M$ control invariant inequality constraints $h_i(\boldsymbol{x}): \mathcal{X}\to \mathbb{R}$, $\forall i \in \{1,...,M\}$ are developed, where $h_i(\boldsymbol{x}) \geq 0$ when the constraint is satisfied. There must exist a control law $\boldsymbol{u}\in \mathcal{U}$ that renders all of the constraints forward invariant. $\mathcal{C}_{\rm S}$ is then defined explicitly as,

$$
    \mathcal{C}_{\rm S} := \{\boldsymbol{x} \in \mathcal{X} \, | \, h_i(\boldsymbol{x}) \geq 0, \forall i \in \{1,...,M\} \}.
$$

The explicit safe set can be determined entirely offline, however this often becomes difficult for higher dimensional and more complex systems. The system can verify that $\boldsymbol{x}\in \mathcal{C}_{\rm S}$ by simply verifying that $h_i(\boldsymbol{x})\geq0, \forall i \in \{1,...,M\}$.

### Implicit Safe Set

$\mathcal{C}_{\rm S}$ can also be defined *implicitly* using closed loop trajectories under a predetermined backup control law $\boldsymbol{u}_{\rm b}$. This backup control law will guide the system to a backup set $\mathcal{C}_{\rm B} \subseteq \mathcal{C}_{\rm S}$, where the system is verified to remain safe for all time. $\mathcal{C}_{\rm S}$ is then defined implicitly as,

$$
    \mathcal{C}_{\rm S} := \{\boldsymbol{x}\in \mathcal{X} \, | \, \forall t\geq 0,\,\, \phi^{\boldsymbol{u}_{\rm b}}(t;\boldsymbol{x})\in\mathcal{C}_{\rm A} \},
$$

where $\phi^{\boldsymbol{u}_{\rm b}}$ represents a prediction of the state $\boldsymbol{x}$ for $t$ seconds under the backup control law $\boldsymbol{u}_{\rm b}$. The implicit safe set is determined entirely online, where backup trajectories must constantly be computed at each time interval. 

In practice, these trajectories are often evaluated over a finite time horizon $\mathcal{T} = [0,T]$ in order to reduce computation time. As a result, a trade-off between computation time and safety guarantees is introduced: as the length of the time horizon decreases, the computation time decreases but safety is guaranteed over a shorter time horizon. As the length of the time horizon increases, the computation time increases while safety is guaranteed over a longer time horizon.

## Simplex RTA Filters

[Simplex](https://apps.dtic.mil/sti/pdfs/ADA307890.pdf) RTA filters switch between the primary controller and a verified backup controller to assure safety of the system. The Simplex filter montiors the desired control input $\boldsymbol{u}_{\rm des}$ from the primary controller, and if $\boldsymbol{u}_{\rm des}$ is determined to be safe, it is passed to the plant unaltered. Otherwise, a safe backup control input $\boldsymbol{u}_{\rm b}$ is instead passed to the plant. For this module, the Simplex RTA filter is designed as follows,

$$
\begin{array}{rl}
\boldsymbol{u}_{\rm safe}(\boldsymbol{x})=
\begin{cases}
\boldsymbol{u}_{\rm des}(\boldsymbol{x}) & {\rm if}\quad \phi_1^{\boldsymbol{u}_{\rm des}}(\boldsymbol{x}) \in \mathcal{C}_{\rm S}  \\ 
\boldsymbol{u}_{\rm b}(\boldsymbol{x})  & {\rm if}\quad otherwise
\end{cases}
\end{array}
$$

Here, $\phi_1^{\boldsymbol{u}_{\rm des}}(\boldsymbol{x})$ represents a prediction of the state $\boldsymbol{x}$ if $\boldsymbol{u}_{\rm des}$ is applied for one discrete time interval $\tau \in \mathcal{T}$. 

### Explicit Simplex RTA

The Simplex filter can be designed using the explicit definition of $\mathcal{C}_{\rm S}$,

$$
    \mathcal{C}_{\rm S} := \{\boldsymbol{x} \in \mathcal{X} \, | \, h_i(\boldsymbol{x}) \geq 0, \forall i \in \{1,...,M\} \}.
$$

For an explicit Simplex RTA, the backup controller should be designed such that $\boldsymbol{u}_{\rm b}$ will always keep the system in $\mathcal{C}_{\rm S}$.

### Implicit Simplex RTA

The Simplex filter can be designed using the implicit definition of $\mathcal{C}_{\rm S}$,

$$
    \mathcal{C}_{\rm S} := \{\boldsymbol{x}\in \mathcal{X} \, | \, \forall t\geq 0,\,\, \phi^{\boldsymbol{u}_{\rm b}}(t;\boldsymbol{x})\in\mathcal{C}_{\rm A} \}.
$$

For an implicit Simplex RTA, the backup controller should be designed using a control law that always leads the system back to $\mathcal{C}_{\rm B}$, which is verified to be safe.

## ASIF RTA Filters

[Active Set Invariance Filter](http://www.ames.caltech.edu/gurriet2018online.pdf) (ASIF) RTA approaches are an optimization-based approach to assuring safety. ASIF approaches solve a qudratic program to minimize the $l^2$ norm difference between $\boldsymbol{u}_{\rm des}$ and $\boldsymbol{u}_{\rm safe}$, where safety is enforced using [control barrier functions](https://arxiv.org/pdf/1903.11199.pdf). This allows the ASIF to be minimally invasive towards the primary controller. For this module, the ASIF RTA is designed as follows,

$$
\begin{split}
\boldsymbol{u}_{\text{safe}}(\boldsymbol{x})={\text{argmin}} & \Vert \boldsymbol{u}_{\text{des}}-\boldsymbol{u}\Vert ^{2} \\
\text{s.t.} \quad & BC_i(\boldsymbol{x},\boldsymbol{u})\geq 0, \quad \forall i \in \{1,...,M\}
\end{split}
$$

Here, $BC_i(\boldsymbol{x},\boldsymbol{u})$ represents one of $M$ barrier constraints, where the barrier constraints are satisfied when $BC_i(\boldsymbol{x},\boldsymbol{u}) \geq 0$. The objective for using barrier constraints is to enforce Nagumo's condition, where the boundary of the set formed by $h_i(\boldsymbol{x})$ is examined to ensure $\dot{h}_i(\boldsymbol{x})$ is never decreasing, and therefore $\boldsymbol{x}$ will never leave $\mathcal{C}_{\rm S}$. For the $i^{th}$ safety constraint, this condition is written as,

$$
    \dot{h}_i(\boldsymbol{x}) = \nabla h_i(\boldsymbol{x}) \dot{\boldsymbol{x}} = L_f h_i(\boldsymbol{x}) + L_g h_i (\boldsymbol{x}) \boldsymbol{u} \geq 0,
$$

where $L_f$ and $L_g$ are Lie derivatives of $h_i$ along $f$ and $g$ respectively. In practice, this condition should only  be applied to the boundary of $\mathcal{C}_{\rm S}$, and not to states far from the boundary. To account for this fact, a strengthening function $\alpha(x)$ is used to relax the constraint away from the boundary. $\alpha(x)$ is a class-$\kappa$ function that is strictly increasing and meets the condition $\alpha(0)=0$. The barrier constraint is then constructed as,

$$
    BC_i(\boldsymbol{x},\boldsymbol{u}) := L_f h_i(\boldsymbol{x}) + L_g h_i (\boldsymbol{x}) \boldsymbol{u} + \alpha(h_i(\boldsymbol{x})).
$$

### Explicit ASIF RTA

For explicit ASIF RTA, each barrier constraint can be constructed explicitly as,

$$
    BC_i(\boldsymbol{x},\boldsymbol{u}) := \nabla h_i(\boldsymbol{x}) (f(\boldsymbol{x}) + g(\boldsymbol{x})\boldsymbol{u}) + \alpha(h_i(\boldsymbol{x})),
$$

where $\boldsymbol{u} \in U$ and $h_{i}(\boldsymbol{x})$ refers to the set of $M$ control invariant safety constraints. 

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

### Implicit ASIF RTA

For implicit ASIF RTA, each barrier constraint can be constructed implicit as,

$$
    BC_i(\boldsymbol{x},\boldsymbol{u}) := \nabla \varphi_i(\phi^{\boldsymbol{u}_{\rm b}}_j) D(\phi^{\boldsymbol{u}_{\rm b}}_j) [f(\boldsymbol{x}) + g(\boldsymbol{x})\boldsymbol{u}] + \alpha(\varphi_i(\phi^{\boldsymbol{u}_{\rm b}}_j) ),
$$

where $\boldsymbol{u} \in U$, $\varphi_i(\boldsymbol{x})$ refers to the set of $M$ safety constraints that define $\mathcal{C}_{\rm A}$, $\phi^{\boldsymbol{u}_{\rm b}}_j$ represents a prediction of the state $\boldsymbol{x}$ at the $j^{th}$ discrete time interval along the backup trajectory $\forall t \in [0,\infty)$, and $D(\phi^{\boldsymbol{u}_{\rm b}}_j)$ is computed by integrating a sensitivity matrix along the backup trajectory. In practice, the trajectory is evaluated over a finite set of $j$ points over the interval $[0,T]$, where a barrier constraint is constructed at each point.

Additionally, rather than constructing a barrier constraint at each point along the backup trajectory, these points can be subsampled such that the barrier constrains are only constructed at the point(s) where $\varphi_i(\phi^{\boldsymbol{u}_{\rm b}}_j)$ is minimized. This allows the ASIF to only consider the most relevant points along the trajectory.