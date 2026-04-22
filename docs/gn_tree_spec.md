# Geometry Nodes Tree Specification — `NeuralNetwork`

Target: Blender 5.1. Node group name: `NeuralNetwork`. Applied as a Geometry Nodes modifier on a single mesh object created by the addon's bootstrap operator.

## Design principles

1. **One node group, flat interface.** All user-tunable parameters live on the Group Input. No nested subgroups unless strictly needed for clarity.
2. **Per-layer logic is symmetric.** Input / L1 / L2 / L3 / Output are all built by the same pattern (grid → trim → instance), so the tree has five copies of the same subgraph with different inputs, joined at the end.
3. **Connections are optional and cheap.** A `Bool` input gates the connection subgraph so users can hide curves entirely for heavy networks.
4. **Aspect is an index switch, not an enum.** `Index Switch` (added Blender 4.1) picks between Icosphere / UV Sphere / Cube meshes by integer. Matches the original AN addon's integer-aspect convention.

## Group interface (sockets)

**Inputs** (in declared order):

| # | Name | Type | Default | Range | Notes |
|---|---|---|---|---|---|
| 1 | Input Size | Int | 3 | 1–4096 | Neuron count for input layer |
| 2 | Input Grid Width | Int | 1 | 1–128 | Columns for input layer (height = ceil(Size/Width)) |
| 3 | L1 Size | Int | 4 | 0–4096 | 0 = skip this layer |
| 4 | L1 Grid Width | Int | 1 | 1–128 | |
| 5 | L2 Size | Int | 0 | 0–4096 | 0 = skip this layer |
| 6 | L2 Grid Width | Int | 1 | 1–128 | |
| 7 | L3 Size | Int | 0 | 0–4096 | 0 = skip this layer |
| 8 | L3 Grid Width | Int | 1 | 1–128 | |
| 9 | Output Size | Int | 2 | 1–4096 | |
| 10 | Output Grid Width | Int | 1 | 1–128 | |
| 11 | Input Aspect | Int | 0 | 0–2 | 0=Ico, 1=Sphere, 2=Cube |
| 12 | Hidden Aspect | Int | 1 | 0–2 | Applies to L1/L2/L3 |
| 13 | Output Aspect | Int | 0 | 0–2 | |
| 14 | Input Mesh Size | Float | 0.3 | 0.01–5.0 | Neuron radius for input |
| 15 | Hidden Mesh Size | Float | 0.3 | 0.01–5.0 | |
| 16 | Output Mesh Size | Float | 0.3 | 0.01–5.0 | |
| 17 | Connection Visibility | Bool | True | — | Master toggle for connection curves |
| 18 | Connection Radius | Float | 0.01 | 0.001–0.5 | Curve radius |
| 19 | Layer Spacing | Float | 5.0 | 0.1–50 | Distance between consecutive layers on X |
| 20 | Neuron Spacing | Float | 1.0 | 0.1–10 | Distance between neurons within a layer (Y/Z) |

**Outputs:**

| # | Name | Type |
|---|---|---|
| 1 | Geometry | Geometry |

## Internal graph — top level

```
Group Input
   │
   ├─► [Build Layer: Input]     ──┐
   ├─► [Build Layer: L1]        ──┤
   ├─► [Build Layer: L2]        ──┼─► [Join Geometry] ──► [Build Connections] ──► Group Output
   ├─► [Build Layer: L3]        ──┤                                    ▲
   └─► [Build Layer: Output]    ──┘                                    │
                                                                    Connection
                                                                    Visibility,
                                                                    Connection
                                                                    Radius
```

## Build-Layer subgraph (applied 5 times, one per layer)

Inputs to each instance: `Size`, `Grid Width`, `Aspect`, `Mesh Size`, `Layer Index` (0–4), `Layer Spacing`, `Neuron Spacing`.

```
Size ─┐
      ├─► [Math: divide] ──► [Math: ceil] ──► grid_height
Width ┘                                            │
                                                    ▼
Grid (Verts X = Grid Width, Verts Y = grid_height,
      Size X = (Grid Width-1) * Neuron Spacing,
      Size Y = (grid_height-1) * Neuron Spacing)
      │
      ▼
[Delete Geometry]  keep if: index < Size
(trims the grid down to exactly Size points by index)
      │
      ▼
[Set Position]  offset = (Layer Index * Layer Spacing, 0, 0)
      │
      ▼
[Instance on Points]
    Instance = [Index Switch]
                 0 → [Mesh Ico Sphere (radius=1, subdiv=2)]
                 1 → [Mesh UV Sphere (radius=1, segs=16, rings=8)]
                 2 → [Mesh Cube (size=1.5)]
               Index = Aspect
    Scale = Mesh Size (vector = Mesh Size on all axes)
      │
      ▼
[Realize Instances]   (so connections can read neuron positions as points)
      │
      ▼
→ Join Geometry (top level)
```

**Skip behavior for hidden layers:** when `L2 Size == 0`, the Delete Geometry step removes all points. The resulting empty geometry is harmless on Join. This keeps the tree structure fixed and avoids conditional branches (GN has no real if/else across geometry; the "empty input" pattern is idiomatic).

## Build-Connections subgraph

Inputs: the joined geometry of all five layers (as points after instance realization — we keep a parallel points-only copy for connection source), `Connection Visibility`, `Connection Radius`.

The challenge: we need to iterate pairs of layers (Input→L1, L1→L2, L2→L3, L3→Output) and draw a curve from every neuron in layer N to every neuron in layer N+1.

**Strategy:** each Build-Layer subgraph also outputs a "points-only" version of itself (the grid points, before Instance on Points). We collect these five point clouds and feed them to a connections builder that knows which pairs to connect.

```
points_input, points_L1, points_L2, points_L3, points_output
       │
       ▼
[For Each Element] (iterating pairs: input→L1, L1→L2, L2→L3, L3→output)
    For each point p_from in layer N:
        [For Each Element] iterating points of layer N+1:
            [Curve Line]  start = p_from.position, end = p_to.position
            [Set Curve Radius]  radius = Connection Radius
       │
       ▼
[Join Geometry]  (all curves together)
       │
       ▼
[Switch]  Geometry, bool = Connection Visibility
       ├── True  → curves joined with layer geometry → Group Output
       └── False → layer geometry only → Group Output
```

**Handling skipped layers:** if `L2 Size == 0`, `points_L2` is empty. The connection pair (L1→L2) produces zero curves, and (L2→L3) produces zero curves. We need to re-route so L1 connects to L3 directly when L2 is skipped. Implementation: maintain a separate `Int` counter per layer via `Compare > 0`; use `Switch` nodes to route points_L1's downstream target to the first non-empty subsequent layer.

**Pragmatic simplification:** for MVP, just let empty layers produce empty connections (L1→L3 direct connection is skipped). The user controls layer presence via size=0; "bridging over" a zero-sized layer is a Phase-4-plus polish concern, not MVP.

## Node count estimate

- 5× Build-Layer (~8 nodes each) = 40
- 1× Connection builder = 10
- Group I/O + Join + Switch = 6
- **Total ~56 nodes.** Above the "script or hand-wire" cutoff Plan agent flagged. **Decision: script it in Python.** See `build_gn_tree.py` (Phase 2).

## Things intentionally NOT in this tree

- **Text neurons** (old aspect 2). Dropped per user decision 2026-04-19.
- **Training visualization** (live activation colors). Stretch goal for post-MVP; would add an Attribute input + Set Material path.
- **Per-neuron color per layer.** Current design: all neurons same material. User can override at the object level.
- **Bias/weight visualization.** Out of scope — neurons and connections only.

## Acceptance criteria for the tree

Given `Input Size=3, L1 Size=4, L2 Size=0, L3 Size=0, Output Size=2, all Grid Widths=1, aspect=sphere, radius=0.3`:
- Must produce exactly 3 + 4 + 2 = **9 instanced neuron meshes**.
- Must produce 3·4 + 4·2 = **20 connection curves** when Connection Visibility=True.
- **0 curves** when Connection Visibility=False.
- Layer centers on X: 0, 5, (skipped), (skipped), 20 (respecting 0-indexed layer position × spacing; skipped layers still occupy their slot on X so layout stays consistent).

These assertions drive `tests/validate_tree.py` in Phase 6.
