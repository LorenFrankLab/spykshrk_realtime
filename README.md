# SPYKSHRK Realtime High Performance Computing for Neuroscience

The goal of this project is to develop a realtime framework for closed loop electrophysiology neuroscience experiments with algorithms accelerated by a cluster. The first version of the framework is designed around rodent navigation tasks with simultaneous tetrode recordings in hippocampal CA1/CA3. Development is broken up into two phases:

1. Simulated hardware: Proof of principle and design of algorithms/experiment
2. Hardware integration: Connections with data acquisition system

Only the first phase is actively planned and being worked on at this time.

## Application / Scope
The primary use case for this system is for designing and running "closed-loop" experiments: testing scientific hypothesis that require realtime computationally intensive analyses that results in experimental manipulations on the order of milliseconds.  This system is well suited to accelerate algorithms that can be parallelized and are adaptive (e.g. uses streaming forms of data).

As a demonstration of how to use the system, it will be designed around specific rodent experiments with animals performing hippocampally dependent mazes. The avaliable datastreams are simultaneous tetrode recordings from hippocampal CA1/CA3 and the animal's position being tracked by a camera.

## Features for Phase 1
1. "Simulation Mode" - simulates experimental hardware for offline testing of system/algorithms.
2. Instrumentation to evaluate system performance, recording program state and timing.
3. Post-simulation analyses tools to aid in the design, developement and evaluation of algorithms.



