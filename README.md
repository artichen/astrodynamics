# Numerical Hydrodynamics: Shock Tubes and Blast Waves

An undergraduate scientific computing project exploring the one-dimensional compressible Euler equations with Python. The notebook implements three explicit numerical schemes and investigates their behaviour in a Sod-type shock-tube problem and a localised blast-wave experiment.

The project connects conservation laws, numerical discretisatio. 

**Author:** Yuhan Chen  
**Tools:** Python, NumPy, Matplotlib and Jupyter

## Project scope

- Represent a compressible gas using density, momentum density and total energy density.
- Construct physical and interface fluxes for explicit time stepping.
- Implement FTCS, Lax-Friedrichs and a two-step Lax-Wendroff formulation.
- Plot density and velocity profiles and compare the shock-tube calculation with supplied reference data.
- Explore numerical smoothing, oscillations and the importance of boundary conditions and time-step selection.



## Physical model

The calculation uses the one-dimensional Cartesian Euler equations for an ideal gas:

$$
\frac{\partial\mathbf{q}}{\partial t}
+\frac{\partial\mathbf{F}(\mathbf{q})}{\partial x}=0,
$$

$$
\mathbf{q}=\begin{pmatrix}\rho\\\rho u\\E\end{pmatrix},
\qquad
\mathbf{F}(\mathbf{q})=
\begin{pmatrix}\rho u\\\rho u^2+p\\u(E+p)\end{pmatrix},
\qquad
p=(\gamma-1)\left(E-\frac{\rho u^2}{2}\right).
$$

Here, $\rho$ is density, $u$ is velocity, $p$ is pressure, $E$ is total energy per unit volume, and $\gamma$ is the ratio of specific heats. The quantities are used in the numerical units of the original exercise; no physical unit system is specified.

For background on these equations, see the [Clawpack Riemann book: Euler equations of gas dynamics](https://www.clawpack.org/riemann_book/html/Euler.html).

## Experiments and methods

| Experiment | Initial conditions | Grid and nominal evolution | Methods used in calculation cells |
| --- | --- | --- | --- |
| Sod-type shock tube | Left: density 2.5, pressure 1.5, velocity 0. Right: density 0.125, pressure 0.1, velocity 0. Discontinuity at x = 0.75; gamma = 1.4. | 150 physical cells plus two ghost cells; dx = 0.01; dt = 0.0003; 100 steps to t = 0.03. | FTCS, Lax-Friedrichs, Lax-Wendroff |
| Localised blast wave | Uniform density 2 and zero velocity. Four central cells have total energy density 60; the surrounding gas has pressure 0.00002. Gamma = 5/3. | 400 physical cells plus two ghost cells; dx = 0.0005; dt = 0.0001; 500 steps to t = 0.05. | Lax-Friedrichs, Lax-Wendroff |

The blast-wave section is labelled "Sedov blast wave" in the notebook. It is a one-dimensional localised blast-wave experiment, without a claim of validation against the spherical Sedov-Taylor solution.


## Files and preparation

Keep the notebook and both data files in the same directory. Before running, rename the included notebook to `hydrodynamics.ipynb` and remove the download suffixes from the data filenames:



## Running the notebook

From the directory containing the files, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib jupyterlab
python -m jupyter lab
```

On Windows, use `.venv\Scripts\activate` for the activation step. Open `hydrodynamics.ipynb` in JupyterLab.

**Run each numerical method from fresh initial conditions.** The calculation cells modify the same state array in place. Clearing plot outputs does not reset that array, and running all method cells consecutively does not produce independent comparisons.

For a shock-tube experiment:

1. Restart the kernel.
2. Run **Imports**, **Grid setup**, **Initial conditions**, **Comparison settings** and **Define hydro functions**.
3. Run only one calculation cell under **Calculating the evolution: Sod shock tube**.
4. Repeat from step 1 for another method.

For a blast-wave experiment:

1. Restart the kernel and run **Imports**.
2. Run **Sedov blast wave: grid**, **Sedov blast wave: initial conditions** and the following hydrodynamics-function definitions.
3. Run one blast-wave calculation cell.
4. Reinitialise the calculation before changing methods.

The two problem sections redefine functions with hard-coded grid sizes. Rerun the appropriate definitions whenever switching problems.

## Known limitations



1. **Reference-time mismatch.** Each fresh shock-tube run advances to t = 0.03, but the plotting cells select `q_exact_03`. Based on the filenames, `q_exact_003` is the intended comparator. Confirm the reference timestamps and use the matching table in every overlay before interpreting agreement.
2. **Boundary indexing.** `range(150)` and `range(400)` omit the rightmost physical cell, making the corresponding final-cell update branches unreachable. The first physical cell also lacks the exterior boundary flux in its update. Copying ghost-cell values does not repair these omissions.
3. **Time-step control.** There is no adaptive time step or positivity check. For the blast initial state, the Courant number `(dt/dx) * max(abs(u) + sqrt(gamma*p/rho))` is approximately 1.15. Time-step suitability must be reassessed as the solution evolves.
4. **Shared mutable arrays.** Flux routines depend on global work arrays, and calculation cells reuse the evolving state. This makes execution order significant.
5. **Unphysical results.** Inspection runs from fresh initial conditions produced negative densities for the shock-tube FTCS and Lax-Wendroff cells, and overflow/non-finite values for the blast-wave Lax-Wendroff cell. These observations concern this implementation and its selected parameters; they do not isolate the numerical method from the boundary and stability issues above.




