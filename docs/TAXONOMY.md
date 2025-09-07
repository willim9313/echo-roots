# Taxonomy & Attribute Framework

## Core Structure
- **A. Classification Skeleton (Taxonomy)**: initial tree for grouping.
- **B. Raw Input Data**: messy, inconsistent product text.
- **C. Normalization Layer**: controlled attributes and value sets.
- **D. Semantic Layer**: candidate graph of ambiguous or user-driven terms.

## Process Flows
- **Initialization**: build draft A & C, seed D.
- **Governance**: merge/split attributes, deprecate unused, elevate from D→C.
- **Semantic Layer Workflow**:
  - D-Intake → D-Cluster → D-Merge/Split → D-Relation Typing → D-Score.
  - Stable terms elevate to C, unstable remain in D, invalid get deprecated.

## Evolution Stages
1. A only (tree).
2. A + C + D combined.
3. C stabilizes, partial D elevation.
4. Full graph: taxonomy + controlled vocab + semantic network.

## Applications
- **E-commerce**: A=product categories, C=brand/color/size, D=consumer terms.
- **Media/Stories**: A=genres, C=directors/roles, D=viewer slang.
- **Gaming, Travel, Food**: flexible cross-domain use cases  .
