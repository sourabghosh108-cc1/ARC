"""Task 1ae2feb7 â€” Indicator->Projection (period-K projection from bars across divider).

STRUCTURE
  Grid has a uniform-color straight-line "divider" (row or column), non-zero
  color (call it D). Cells on one side of divider contain "bars": horizontal
  (if divider is column) or vertical (if divider is row) contiguous runs of
  non-background color. Bars lie on rows/cols perpendicular to divider.
  Other side is empty (all background).

INDICATOR
  For each row (or col) perpendicular to the divider, scan FROM divider
  OUTWARD on the bar side. Identify contiguous same-color runs. Each run is
  a bar with: length K, color C, rank r (1=closest to divider, 2=next, ...).

PROJECTION
  For each row (or col), on the empty side starting adjacent to divider,
  overlay period-K patterns for each bar (place color C at offset 0 mod K,
  zero elsewhere). Closer bars (lower rank) take priority over further bars.

ALGORITHM
  1. Detect divider line (row or column of uniform non-bg color).
  2. For each perp line (row if vertical divider, col if horizontal):
       a. Scan from divider cell outward on the bar side.
       b. Collect contiguous runs of same non-zero color as bars.
       c. On the empty side, for each position p (distance d from divider,
          d=1, 2, ...), compute color: iterate bars from closest; first bar
          whose period K matches (d-1 % K == 0) sets color C; else 0.
"""


def _detect_divider(grid):
    """Return ('col', col_idx, D) or ('row', row_idx, D)."""
    H, W = len(grid), len(grid[0])
    # Try columns
    for c in range(W):
        vals = set(grid[r][c] for r in range(H) if grid[r][c] != 0)
        col_vals = [grid[r][c] for r in range(H)]
        # divider = a non-zero color forming most of the column
        from collections import Counter
        cnt = Counter(col_vals)
        non0 = [(v, n) for v, n in cnt.items() if v != 0]
        if non0:
            val, n = max(non0, key=lambda x: x[1])
            if n >= H * 0.5 and len(set(col_vals) - {0, val}) <= 2:
                # Check this is really a divider (column mostly val)
                if n >= H - 2:  # allow 1-2 gaps (like test[0] row 9)
                    return ('col', c, val)
    # Try rows
    for r in range(H):
        row_vals = list(grid[r])
        from collections import Counter
        cnt = Counter(row_vals)
        non0 = [(v, n) for v, n in cnt.items() if v != 0]
        if non0:
            val, n = max(non0, key=lambda x: x[1])
            if n >= W - 2:
                return ('row', r, val)
    return None


def _bars_in_line(line, divider_idx, bar_side):
    """Given a 1D line and divider index, scan from divider outward on bar_side.
    bar_side: -1 (left/up) or +1 (right/down).
    Return list of bars [(K, C, start_offset_from_divider)] in rank order.
    """
    bars = []
    i = divider_idx + bar_side
    current_color = None
    current_len = 0
    while 0 <= i < len(line):
        v = line[i]
        if v == 0:
            if current_color is not None and current_color != 0:
                bars.append((current_len, current_color))
            current_color = 0
            current_len = 0
        else:
            if v == current_color:
                current_len += 1
            else:
                if current_color is not None and current_color != 0:
                    bars.append((current_len, current_color))
                current_color = v
                current_len = 1
        i += bar_side
    if current_color is not None and current_color != 0:
        bars.append((current_len, current_color))
    return bars


def _project_line(line, divider_idx, empty_side, bars):
    """Fill the empty side with overlay of period-K patterns from bars.
    bars is list of (K, C) in rank order (closest first).
    empty_side: +1 or -1.
    """
    i = divider_idx + empty_side
    d = 1  # distance from divider (1-indexed)
    while 0 <= i < len(line):
        # Determine color for this position
        offset0 = d - 1  # 0-indexed offset on empty side
        color = 0
        for (K, C) in bars:
            if offset0 % K == 0:
                color = C
                break
        line[i] = color
        i += empty_side
        d += 1


def solve(grid):
    grid = [list(row) for row in grid]
    H, W = len(grid), len(grid[0])
    det = _detect_divider(grid)
    if det is None:
        return grid
    kind, idx, D = det

    if kind == 'col':
        for r in range(H):
            line = grid[r]
            # Determine bar side: side with more non-zero, non-D cells
            left_nz = sum(1 for c in range(idx) if line[c] != 0)
            right_nz = sum(1 for c in range(idx + 1, W) if line[c] != 0)
            if left_nz == 0 and right_nz == 0:
                continue
            if left_nz >= right_nz:
                bar_side = -1
                empty_side = +1
            else:
                bar_side = +1
                empty_side = -1
            bars = _bars_in_line(line, idx, bar_side)
            if bars:
                _project_line(line, idx, empty_side, bars)
    else:  # row divider
        for c in range(W):
            line = [grid[r][c] for r in range(H)]
            top_nz = sum(1 for r in range(idx) if line[r] != 0)
            bot_nz = sum(1 for r in range(idx + 1, H) if line[r] != 0)
            if top_nz == 0 and bot_nz == 0:
                continue
            if top_nz >= bot_nz:
                bar_side = -1
                empty_side = +1
            else:
                bar_side = +1
                empty_side = -1
            bars = _bars_in_line(line, idx, bar_side)
            if bars:
                _project_line(line, idx, empty_side, bars)
                for r in range(H):
                    grid[r][c] = line[r]
    return grid
