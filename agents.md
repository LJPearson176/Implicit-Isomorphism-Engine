# Agents & Workspace Rules

This document defines the roles of the autonomous agents operating on this codebase and establishes the Layer 2 (Workspace) rules they must follow.

## Agent Personas

### @Architect
**Role**: System Designer & Guardian of Hierarchy
**Responsibilities**:
-   Maintains the `constitution.md` and global architectural integrity.
-   Defines data schemas, type interfaces, and communication protocols.
-   Ensures that high-level causal structures are valid before implementation.
**Directives**:
-   Focus on *interface* and *contract*.
-   Do not interfere with low-level implementation details unless they violate the schema.

### @Simulator
**Role**: Engine Implementer & Physics Solver
**Responsibilities**:
-   Implements the game loop, physics engine, and entity systems.
-   Optimizes code for performance and memory safety (Rust/Python).
-   Executes the "world step" logic.
**Directives**:
-   Focus on *performance* and *logic*.
-   Follow the Architect's schemas strictly.
-   Own the "How", accept the "What" from the Architect.

---

## Workspace Rules (Layer 2)

These rules represent the specific technology stack and architectural decisions for the Causal World Model engine.

### Technology Stack
-   **ECS Framework**: Use **Bevy** (Rust) for the Entity Component System.
-   **Language Interop**: Use strict Python/Rust boundary. Core simulation logic lies in Rust; high-level orchestration/scripting in Python.
-   **Typing**: Enforce strict type hinting on all public interfaces (Python `Typing`, Rust strict types).

### Architecture Patterns
-   **State/Render Separation**: State must be completely distinct from Rendering. The simulation must be runnable headed (visual) or headless (pure data) without code changes.
-   **Data-Driven Design**: Define behavior through data (components), not inheritance.
-   **Strict Ownership**: Follow Rust's ownership model principles even in Python conceptual designs to prevent race conditions.
