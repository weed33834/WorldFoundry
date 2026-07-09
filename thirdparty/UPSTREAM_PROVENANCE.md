# Third-Party Upstream Provenance

This file records the local provenance review for vendored code under `thirdparty/`.
Entries are based only on files present in this repository.

## `simple-knn` (modified fork)

- upstream_url: `https://github.com/camenduru/simple-knn`
- local_path: `thirdparty/simple-knn`
- fork_status: **modified** — not the unmodified upstream. See `MODIFICATIONS.md` for details.
- modifications: Build configuration and CUDA kernels adapted for WorldFoundry integration and compatibility with the depth-modified `diff-gaussian-rasterization` fork.
- evidence: Source headers identify Inria GRAPHDECO copyright.
- purpose: CUDA extension for average nearest-neighbor distance over 3D points.
- license_summary: see `thirdparty/THIRD_PARTY_LICENSES.md`.

## `diff-gaussian-rasterization` (modified fork — depth output)

- upstream_url: `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/`
- upstream_repo: `https://github.com/graphdeco/diff-gaussian-rasterization`
- local_path: `thirdparty/diff-gaussian-rasterization`
- fork_status: **modified** — not the unmodified upstream. See `MODIFICATIONS.md` for details.
- modifications: Forward pass returns `depth` alongside `color` and `radii`; backward pass propagates depth gradients; CUDA kernels extended for depth rendering.
- evidence: Source headers identify Inria GRAPHDECO copyright; `MODIFICATIONS.md` documents changes.
- purpose: CUDA rasterization extension for 3D Gaussian Splatting with depth rendering, used by pixelSplat and other WorldFoundry models requiring depth maps.
- nested_third_party:
  - `third_party/stbi_image_write.h`: single-header image writer from stb.
  - `third_party/glm`: listed in `.gitmodules` as `https://github.com/g-truc/glm.git`; GLM source files are present in the current tree.
- license_summary: see `thirdparty/THIRD_PARTY_LICENSES.md`.

## `depth-diff-gaussian-rasterization-min` (modified fork — extended depth outputs)

- upstream_url: `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/`
- upstream_repo: `https://github.com/graphdeco/diff-gaussian-rasterization`
- local_path: `thirdparty/depth-diff-gaussian-rasterization-min`
- fork_status: **heavily modified** — not the unmodified upstream. See `MODIFICATIONS.md` for details.
- modifications: Forward pass returns `depth`, `median_depth`, and `final_opacity` alongside `color` and `radii`; CUDA kernels extended for all additional outputs.
- evidence: Source headers identify Inria GRAPHDECO copyright; `MODIFICATIONS.md` documents changes.
- purpose: Minimal depth-focused variant for models requiring detailed depth information (median depth, final opacity) for evaluation metrics.
- license_summary: see `thirdparty/THIRD_PARTY_LICENSES.md`.

## `gsplat`

- upstream_url: `https://github.com/nerfstudio-project/gsplat.git`
- local_path: `thirdparty/gsplat`
- source_commit: `b5392febf6047655c18db17693636cd21bbe58c0`
- evidence: shallow clone HEAD recorded in `.worldfoundry_upstream_commit`; upstream `LICENSE` is retained locally.
- purpose: CUDA accelerated Gaussian splatting rasterization with Python bindings.
- nested_third_party:
  - `gsplat/cuda/csrc/third_party/glm`: listed in upstream `.gitmodules` as `https://github.com/g-truc/glm.git`.
- license_summary: see `thirdparty/THIRD_PARTY_LICENSES.md`.
