# ICE Runtime

[![ICE Ecosystem](https://img.shields.io/badge/ICE-Ecosystem-8FB9FF?style=flat)](#)
[![Docs](https://img.shields.io/badge/docs-ICE%20Docs-8FB9FF?style=flat)](https://github.com/francescomaiomascio/ice-docs)
[![Status](https://img.shields.io/badge/status-active%20development-6B7280?style=flat)](#)
[![Language](https://img.shields.io/badge/python-3.x-111827?style=flat)](#)
[![License](https://img.shields.io/badge/license-MIT-7A7CFF?style=flat)](#)

ICE Runtime is the **execution core** of the ICE ecosystem.

It provides a structured, extensible, and policy-aware runtime designed to
coordinate applications, sessions, events, memory, and transports in a
coherent and inspectable system.

ICE Runtime is not a traditional framework.
It is an **operational substrate** for long-living, stateful, and observable
intelligent systems.

---

## Role in the ICE Ecosystem

ICE Runtime sits at the center of ICE execution.

It does **not** define intelligence, domain logic, or user interfaces.
Instead, it enforces **how things happen**, **in which order**, and **under which rules**.

All ICE applications, agents, and products ultimately execute *through* the Runtime.

---

## Core Principles

- Explicit lifecycle management  
- Strong separation of responsibilities  
- Event-driven coordination  
- Capability-based access control  
- Deterministic state transitions  
- Runtime introspection and observability  

---

## Responsibilities

ICE Runtime is responsible for:

- Managing process, session, and workspace lifecycles
- Executing and supervising runs
- Routing, validating, and persisting events
- Enforcing capability and authority boundaries
- Governing state transitions and memory exposure
- Providing structured logging and transport abstractions
- Acting as the execution substrate for higher-level ICE systems

---

## Usage

ICE Runtime is typically **not consumed directly by end users**.

It is embedded or orchestrated by:

- Agent systems
- IDE integrations
- Automation layers
- ICE products such as ICE Studio

A minimal entrypoint is available via:

```bash
python -m ice_runtime
