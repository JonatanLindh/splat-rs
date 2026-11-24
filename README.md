# splat-rs

A real-time 3D Gaussian Splatting viewer written from scratch in Rust and [wgpu](https://wgpu.rs). It loads scenes trained with the original [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) code and renders them at **100+ FPS at 4K** on the standard pre-trained *Truck* and *Train* scenes (AMD RX 9070 XT).

## How it works

Each frame runs entirely on the GPU, with no CPU readback:

1. **Cull and compact** (compute): every splat is tested against the view frustum. Visible splats are compacted with subgroup prefix sums plus one atomic per subgroup, and their view-space depth is written as a sortable `u32` key.
2. **Indirect arguments** (compute): the visible count is turned into the dispatch arguments for the sort and the draw arguments for rendering, so the CPU never needs to know how many splats survived.
3. **Depth sort** (compute): a 4-pass, 8-bit LSD radix sort (count → scan → scatter, ping-ponging between buffers) orders the visible splats by depth.
4. **Rasterize** (render): one instanced quad per splat. The vertex shader projects each 3D Gaussian to a 2D covariance (EWA splatting: Σ₂D = J W Σ Wᵀ Jᵀ) and evaluates view-dependent colour from spherical harmonics up to degree 3. The fragment shader evaluates the Gaussian and discards negligible contributions.

Two transparency modes are available:

- **Alpha blending** of the depth-sorted splats (the standard 3DGS approach).
- **Stochastic transparency** with MSAA and a depth buffer, with a selectable sample count.

## Features

- Loads binary 3DGS `.ply` files with any spherical harmonics degree (0–3).
- Free-fly camera and an egui side panel with a file picker, rendering options and a culling debug view.
- WGSL shaders with modular imports (`naga_oil`), and Rust bindings generated at build time (`wgsl_bindgen`), so the pipeline layouts are type-checked.
- Shader hot-reloading in debug builds.
- A parallel CPU radix sort (Rayon) used as a reference.

## Running

```sh
cargo run --release
```

Open a `.ply` file from the side panel. The viewer also loads `point_cloud.ply` or `train.ply` from the working directory if one exists. The pre-trained scenes are available from the [3DGS project page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

**Controls:** right-drag to look around, `W`/`A`/`S`/`D` to move, `E`/`Space` up, `Q`/`Shift` down, scroll to move forward/back.

**Requirements:** a GPU and driver with subgroup operations and push constants (immediates). Developed on Linux with Vulkan. A Nix flake provides a dev shell (`nix develop`).
