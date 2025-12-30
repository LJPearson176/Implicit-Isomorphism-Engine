# Global Constitution

## 1. The Priority Clause
> **Critical Rule**: If instructions conflict, follow this order:
> 1.  **Current User Request** (Highest Priority)
> 2.  **Workspace Rules** (Project-Specific Stack & Standards)
> 3.  **Global Constitution** (Safety, Causality, Idempotency)

---

## 2. The Context Engineering Hierarchy

To ensure conflict-free collaboration between agents, we adhere to the following layered governance model:

### Layer 1: Global Constitution (The Foundation)
**Scope**: Universal principles of Safety, Causality, and Idempotency.
**Authority**: Immutable baseline unless overridden by higher layers.

### Layer 2: Workspace Rules (The Stack)
**Scope**: Project-specific technology choices and architectural patterns.
**Authority**: Defines *how* we implement the Causal World Model (e.g., Bevy ECS, Python/Rust bridge).
**Location**: Defined in `agents.md` and workspace configuration.

### Layer 3: Workflows (On-Demand)
**Scope**: Ephemeral scripts and processes.
**Authority**: Operational tasks (e.g., "Stress Test Physics", "Validate History").

---

## 3. Core Principles (Layer 1)

### Safety
-   **Ensure Data Integrity**: Always validate input data against strict schemas before processing.
-   **Maintain Isolation**: Keep experimental simulation branches isolated from the main production state.
-   **Explicit Failure Handling**: Handle all potential errors explicitly; ensure the system fails safely and predictably.

### Causality
-   **Enforce Temporal Consistency**: Ensure that all events strictly follow a definition of "cause precedes effect."
-   **Track Provenance**: Maintain a comprehensive log of the origin and transformation history for all entity states.
-   **Deterministic State Transitions**: Guarantee that given the same initial state and input, the next state is always identical.

### Idempotency
-   **Repeatable Operations**: Design all state-modifying functions to produce the same result regardless of how many times they are executed.
-   **Atomic Transactions**: Ensure that complex updates either succeed completely or result in no change at all.
-   **Stateless Execution**: Prefer stateless logic where possible; pass full context explicitly to functions.
