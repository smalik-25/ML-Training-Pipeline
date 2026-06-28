"""Generate small, schema-faithful synthetic Parquet fixtures.

Every downstream stage and all of CI run against these fixtures -- no live
Postgres required. Fixtures must be statistically plausible (realistic premium
distributions, enough sales history per shoe that rolling_7d_avg_premium
populates), otherwise Phase 3's Pandera statistical checks would be theater.
One deliberately-bad fixture is also produced for the validation failure test.

Implemented in Phase 1. Writes to data/fixtures/.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Fixture generation is implemented in Phase 1.")


if __name__ == "__main__":
    main()
