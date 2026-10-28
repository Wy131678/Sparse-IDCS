# Upstream baseline provenance

Source repository: https://github.com/ShiLevy/IDCS

Upstream paper: Levy, S., Friedli, L., Mariethoz, G., and Linde, N. (2024). Conditioning of multiple-point statistics simulations to indirect geophysical data. *Computers and Geosciences*, 187, 105581. https://doi.org/10.1016/j.cageo.2024.105581.

Licence: GNU General Public License version 3.0. The copied Python modules in this directory are retained as the baseline core. The release runner uses only its linear straight-ray path. The local wrapper in `src/sparse_idcs/research.py` supplies the paper geometry and compatibility shims; it should not be represented as the unmodified upstream command-line interface.
