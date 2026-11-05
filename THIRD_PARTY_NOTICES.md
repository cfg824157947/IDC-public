# Third-party dependency license review

IDC's original source is released under MIT. Review date: 2026-11-04.
No third-party source implementations, dependency wheels, shared libraries, model
weights, or gene-set databases are bundled in this source release. Scientific
formula implementations (including NB and DL) are project source, not copied
DESeq2 or scran code. An implementation inspired by a statistical procedure is
not a redistribution of that package.

The declared dependencies were checked against the installed release metadata
and license files; upstream NumPy/SciPy and PyTorch licensing was cross-checked.
All listed project licenses permit use by an MIT-licensed IDC source package.
Dependency copyright and license terms remain in force; IDC's MIT license does
not relicense dependencies or their bundled subcomponents.

| Dependency | Tested version | Project license |
|---|---|---|
| numpy | 2.2.6 | BSD-3-Clause |
| scipy | 1.15.2 | BSD-3-Clause |
| pandas | 2.3.3 | BSD-3-Clause |
| scikit-learn | 1.7.2 | BSD-3-Clause |
| anndata | 0.11.4 | BSD-3-Clause |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| pytest | 9.0.3 | MIT |
| torch | 2.5.1 | BSD-3-Clause |
| setuptools | 82.0.1 | MIT |
| wheel | 0.46.3 | MIT |

Core runtime: NumPy, SciPy, pandas, scikit-learn, AnnData, threadpoolctl, PyYAML.
PyTorch is optional GPU support; pytest is optional testing; setuptools/wheel are
build tools. These dependencies and transitive libraries are installed separately.
Their full notices travel with their own distributions. NumPy/SciPy/PyTorch
binary distributions may include additional licenses for bundled libraries;
this audit approves IDC's source-only artifact, not redistribution of an entire
Python/CUDA environment or every future dependency version.

License references:
- https://opensource.org/license/mit
- https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt
- https://github.com/scipy/scipy/blob/v1.15.2/LICENSE.txt
- https://github.com/pandas-dev/pandas/blob/v2.3.3/LICENSE
- https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING
- https://github.com/scverse/anndata/blob/0.11.4/LICENSE
- https://github.com/joblib/threadpoolctl/blob/3.6.0/LICENSE
- https://github.com/yaml/pyyaml/blob/6.0.3/LICENSE
- https://github.com/pytest-dev/pytest/blob/main/LICENSE
- https://github.com/pytorch/pytorch/blob/v2.5.1/LICENSE

No conflicting third-party notice or vendored implementation was found in the
included project files. Source origins and content hashes are retained in
provenance/sources.json using project-relative paths.
