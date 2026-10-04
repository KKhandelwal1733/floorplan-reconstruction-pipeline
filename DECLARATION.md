# Fix Loop Declaration

## Issue

The ceiling-detection heuristic (`roomscan/geometry/planes.py::find_floor_ceiling`,
`_ceiling_fill_ratio`) rejects a genuine ceiling in the real `real_with_ceiling`
fixture, reporting `ceiling_unobserved=True` (wide-CI abstain fallback) instead of a
measured ceiling height.

This was found and documented honestly during Phase 6's real-data verification (see
project memory `feedback_ceiling_fill_ratio.md`): the real ceiling candidate has a
physically plausible room-height estimate (~2.4 m) but only a 0.23 2-D fill ratio —
too close to a synthetic wall-top false positive's 0.28-0.29 fill ratio to safely
loosen the existing `CEIL_MIN_FILL_RATIO=0.35` threshold without reintroducing the
false-positive bug that threshold exists to prevent. At the time, the conservative
choice (keep abstaining) was made deliberately, per hard rule #2 (abstain/widen over
confident garbage), and documented as a known calibration gap to revisit.

## Root cause

`_ceiling_fill_ratio` measures a single signal: what fraction of a coarse 2-D grid's
cells are touched by inlier points, anywhere in the candidate plane's bounding box.
A real ceiling swept patchily by a handheld phone and a ring of wall-top points
(which only exist around the room's perimeter) can produce similar-looking global
fill ratios, because the metric doesn't distinguish *where in the room* those filled
cells are.

## Proposed fix

Add a second, independent signal: the fraction of inlier points falling in the
*interior* of the candidate plane's own bounding box (a margin-shrunk inner
rectangle), not just anywhere in it. A wall-top ring, by construction, can never
have points in the room's interior — only at the walls around the edge. A real
ceiling, even incompletely swept, is reached while walking around *inside* the room,
so some meaningful fraction of its points should fall in the interior.

Quick empirical check (not yet wired into the codebase — this declaration is the
"before" state) on the two cases already on file:

| Case | fill_ratio (existing) | interior_frac (proposed) |
|---|---|---|
| Synthetic wall-top false positive (3 planes) | 0.284, 0.284, 0.290 | **0.000, 0.000, 0.000** |
| Real `real_with_ceiling` genuine ceiling | 0.229 | **0.719** |

The new metric separates the two cases by a wide margin (0.000 vs 0.6-0.73), unlike
fill ratio (0.23 vs 0.28-0.29, too close to trust). This check will be captured as a
reproducible script (`bench/`) alongside the implementation, not left as an
ad hoc one-off.

## Success criteria

1. `real_with_ceiling` detects a ceiling plane (`ceiling_unobserved=False`) with a
   plausible room-height value, using representative frame-stride sampling as
   established in Phase 5/6 (a chronological prefix is known to be misleading for
   this fixture; see project memory).
2. The existing synthetic wall-top rejection test
   (`tests/test_ceiling_missing.py`, and the wall-top case this declaration's table
   reproduces) still correctly rejects the false positive -- this fix must not
   trade one failure mode for another.
3. Full test suite still green.
4. If criterion 1 is not met after implementation (possible -- real-world
   calibration sometimes doesn't transfer as cleanly as a two-point check
   suggests), this will be reported honestly as a partial result, not hidden.

## Process

- This file is committed alone, before any fix code is written ("before" evidence).
- The fix will be implemented and tested next.
- A second commit will record the "after" state: whether criterion 1 was met, the
  real before/after numbers, and the final test suite status.
