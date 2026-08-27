"""Task b99e7126 â€” Indicator->Projection (shape completion on cell grid).

STRUCTURE
  N x N input: grid of 3x3 "cells" separated by background-color rows/cols.
  Most cells share a default pattern ("normal"). A few are "special" â€” same
  3x3 footprint but with a different internal marker color.

INDICATOR
  The 3x3 pattern INSIDE each special cell IS the template shape. Specifically,
  the positions where the MARKER color appears (marker = color in special but
  not in normal) form the template's 3x3 mask.

PROJECTION
  Positions of special cells on the cell grid form a PARTIAL copy of the
  template. Find the offset (dr,dc) such that input positions are a subset of
  (template + offset). Fill template positions not in input with new specials.
"""
from collections import Counter


def _cell_layout(grid):
    H = len(grid); W = len(grid[0])
    row_is_sep = [len(set(grid[r])) == 1 for r in range(H)]
    col_is_sep = [len(set(grid[r][c] for r in range(H))) == 1 for c in range(W)]
    cell_rows = []
    r = 0
    while r < H:
        if row_is_sep[r]:
            r += 1; continue
        if r + 3 <= H and not any(row_is_sep[r + k] for k in range(3)):
            cell_rows.append((r, r + 3)); r += 3
        else:
            r += 1
    cell_cols = []
    c = 0
    while c < W:
        if col_is_sep[c]:
            c += 1; continue
        if c + 3 <= W and not any(col_is_sep[c + k] for k in range(3)):
            cell_cols.append((c, c + 3)); c += 3
        else:
            c += 1
    return grid[0][0], cell_rows, cell_cols


def _cell(grid, r0, r1, c0, c1):
    return tuple(tuple(grid[r][c] for c in range(c0, c1)) for r in range(r0, r1))


def solve(grid):
    grid = [list(row) for row in grid]
    _, cell_rows, cell_cols = _cell_layout(grid)
    cells = {}
    for ci, (r0, r1) in enumerate(cell_rows):
        for cj, (c0, c1) in enumerate(cell_cols):
            cells[(ci, cj)] = _cell(grid, r0, r1, c0, c1)
    normal = Counter(cells.values()).most_common(1)[0][0]
    special_positions = [k for k, v in cells.items() if v != normal]
    if not special_positions:
        return grid
    special_cell = cells[special_positions[0]]
    normal_colors = set(c for row in normal for c in row)
    special_colors = set(c for row in special_cell for c in row)
    marker_cand = special_colors - normal_colors
    if marker_cand:
        marker = next(iter(marker_cand))
    else:
        # count differences
        dc = Counter()
        for r in range(3):
            for c in range(3):
                if special_cell[r][c] != normal[r][c]:
                    dc[special_cell[r][c]] += 1
        if not dc:
            return grid
        marker = dc.most_common(1)[0][0]
    template = {(r, c) for r in range(3) for c in range(3) if special_cell[r][c] == marker}
    input_set = set(special_positions)
    n_rows = len(cell_rows); n_cols = len(cell_cols)
    best = None
    for dr in range(-2, n_rows):
        for dc in range(-2, n_cols):
            projected = {(tr + dr, tc + dc) for tr, tc in template}
            if input_set.issubset(projected) and all(
                0 <= pr < n_rows and 0 <= pc < n_cols for pr, pc in projected
            ):
                extra = projected - input_set
                if best is None or len(extra) < len(best[1]):
                    best = ((dr, dc), extra)
    if best is None:
        return grid
    _, extras = best
    for (pr, pc) in extras:
        r0, r1 = cell_rows[pr]
        c0, c1 = cell_cols[pc]
        for r in range(r0, r1):
            for c in range(c0, c1):
                grid[r][c] = special_cell[r - r0][c - c0]
    return grid
