# msr

Multi-superquadric recovery research workspace.

## Structure

- `external/EMS-superquadric_fitting/`: third-party EMS implementation (MIT)
- `data/`: datasets and generated artifacts
- `scripts/`: download/conversion/experiment scripts
- `src/`: core code for your own optimization and recovery methods
- `docs/`: notes and method summaries
- `PDFs/`: papers and references
- `THIRD_PARTY_NOTICES.md`: third-party attribution

## Quick start

1. Create/activate conda env: `msr_ems_opt`
2. Install local package (if needed):
   - `pip install -e external/EMS-superquadric_fitting/Python`
3. Run EMS tests from:
   - `external/EMS-superquadric_fitting/Python/tests`
