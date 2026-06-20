# LISFLOOD-FP FV1-CUDA large-grid patches

The paper's 10 m simulation has ~10.6 million cells (2440 × 4362), which exceeds the
limits of the stock LISFLOOD-FP v8.2 FV1-CUDA solver. Three patches make the
convection-permitting run feasible on a single NVIDIA A100-40 GB:

1. **Padded memory layout** — pad the device arrays so row pitch matches the CUDA
   alignment requirement, removing the out-of-bounds access on large domains.
2. **Padded pinned-buffer host↔device transfer** — match the host pinned buffers to the
   padded device layout to avoid `cudaErrorIllegalAddress` on the rain/output copies.
3. **2-D kernel launch** — replace the 1-D grid/block launch (which overflows the
   maximum grid dimension at this cell count) with a 2-D launch covering the full domain.

## How to add the patch files to this repository

These patches are applied to the build used for the paper, located on the HPC at
`LISFLOOD_CUDA_build/LISFLOOD-FP/`. To release them as portable diffs against the
public v8.2 source:

```bash
# from the modified LISFLOOD-FP build directory on the HPC
git diff v8.2 -- <modified source files>  > fv1_cuda_largegrid.patch
# or, if not under git, diff against a clean v8.2 checkout:
diff -u v8.2_clean/ LISFLOOD-FP/        > fv1_cuda_largegrid.patch
```

Then place the resulting `.patch` file(s) in this directory and apply with:

```bash
cd LISFLOOD-FP-8.2 && git apply fv1_cuda_largegrid.patch && make
```

LISFLOOD-FP v8.2 (unpatched) is archived at https://zenodo.org/records/13121102.
The patches are derivative of that source and inherit its licence.

> **TODO (author):** drop the generated `fv1_cuda_largegrid.patch` here before
> minting the Zenodo DOI.
