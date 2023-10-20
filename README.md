# Exactly modeling the unitary dynamics of quantum interfaces with collision models
## Bruno Ortega Goes, M.Sc.

This repository is dedicated to the comprehensive collection of codes utilized to generate the plots and/or analytical expressions showcased within the pages of my Ph.D. thesis. I use either Python, which utilizes [qutip](https://qutip.org/) or Mathematica, where I use [Melt!](https://melt1.notion.site/). 


[Chapter 1: Quantum measurements](Chapter1)

In this chapter, I discuss the von Neumann measurement model. With the aid of a simple toy model where we have a target system with an arbitrary spin interacting with a single bosonic mode. It introduces several important concepts such as the target and meter system, pre-measurement, collapse, and the figures of merit used to benchmark the steps of the measurement. 

[Chapter 2: Closed-system solution of the 1D atom from collision model](Chapter2)

In this chapter, we provide a comprehensive review and detailed analysis of the findings presented in Ref. [Entropy 2022, 24(2), 151](https://www.mdpi.com/1099-4300/24/2/151). These results serve as the foundation for the subsequent investigations conducted in Refs. [arXiv:2205.09623](https://arxiv.org/abs/2205.09623) and [Phys. Rev. A 107, 023710](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.023710), as well as the forthcoming chapters of this thesis. Our focus is on the closed dynamics of a two-level system, also referred to as a 2LS or qubit, interacting with a field. Through solving this interaction, we derive analytical expressions for the joint wave function of the qubit and the field under two specific initial states: a coherent state and a single photon.

[Chapter 3: Quantum weak values and Wigner negativity in a single qubit gate](Chapter3)

This chapter is based on the results published in [Phys. Rev. A 107, 023710](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.023710). Our findings highlight an intriguing phenomenon: when a qubit interacts with a strong resonant coherent field, alongside post-selection, we observe the emergence of anomalous weak values for the field's excitations. Additionally, we discover that the associated field state exhibits Wigner function negativity, adding further intrigue to this exciting study.

[Chapter 4: The spin-photon interface: Energy-efficient quantum non-demolition measurement and photon-photon gate proposal](Chapter4)

In this chapter, we investigate the measurement of the spin state in a degenerate four-level system (4LS) with linear optical selection rules. The goal is to achieve a non-destructive measurement of the spin state with limited energy resources. The system interacts with fields of different natures: a resonant coherent field
with Poissonian statistics and a superposition of vacuum and a single photon with sub-Poissonian statistics. The restricted energy budget, where both fields have at most one photon on average, poses a challenge. We explore the potential quantum advantage in the pre-measurement process, where entanglement is generated between the field and the
4LS, and in the collapse step, where the light state is measured to infer the spin state. These steps are crucial for key quantum information protocols, such as the generation of cluster states. Finally, we provide an application proposing a single-rail photon-photon gate, which is a two-qubit photonic gate based on the absence (vacuum) or presence
(single photon) of a photon as the logical basis. We consider the ideal scenario where the main imperfection arises from the change in the temporal shape of the scattered field. By utilizing the solutions obtained in Chapter 2, we derive an analytical expression to account for this imperfection. We characterize
the error by computing the state-averaged fidelity and the error process matrix for the photon-photon gate. The findings in this chapter are based on results reported in Ref. [Quantum](https://quantum-journal.org/papers/q-2023-08-31-1099/).

[Chapter 5: The spin-photon interface subjected to an in-plane magnetic field](Chapter5)

In this chapter, we focus on analyzing the spin dynamics in the LRP under realistic experimental conditions of Ref. [Nat. Photon. 17, 582–587 (2023)](https://www.nature.com/articles/s41566-023-01186-0).
We consider a system where the ground state consists of electron spins and the excited state comprises trions, both influenced by the magnetic
field with characteristic Larmor frequencies: $\Omega_{g}$ and $\Omega_{e}$, respectively. The presence of parasitic fields affecting the electron
spin introduces an inherent physical difference between these frequencies, which affects the fidelity of spin rotations. Another significant complication is the necessity to target a moving spin state with a pulse of light. This dynamic nature introduces complexities in accurately timing the pulse delivery.

This chapter aims to address these intrinsic imperfections, i.e., imperfections that are not caused by noise or decoherence, and provide insights into the performance of the LRP by applying a collisional model to a realistic experimental scenario. Our analytical solution enables us to investigate important performance measures, such as the fidelity of spin rotations.

Acknowledgments: This project has received funding from the European Union’s Horizon 2020 Research and Innovation Programme under the Marie Skłodowska-Curie Grant Agreement No. 861097.



