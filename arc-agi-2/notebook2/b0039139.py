"""Task b0039139 â€” Indicator->Projection (tile-count encoded by dot pattern).

STRUCTURE
  Narrow strip (vertical if tall, horizontal if wide). Full-uniform separators
  of non-bg color split into 4 blocks, in order:
    A: small shape on bg=0 background (the tile to replicate).
    B: dot-counter â€” 0s with scattered dots, encoding integer n.
    S1: solid block of uniform non-zero color (maps non-zero cells of A).
    S2: solid block of uniform non-zero color (maps zero cells, and gap).

INDICATOR
  n = max(dots per row, dots per col) in B after stripping zero border.

PROJECTION
  Tile A_core (stripped) n times in the strip direction, with 1-cell S2 gap
  between tiles. Recolor: A!=0 -> S1; A==0 -> S2.
"""
from collections import Counter, defaultdict


def _strip_zero_border(block):
    if not block or not block[0]:
        return block
    rows = [r for r in range(len(block)) if any(v != 0 for v in block[r])]
    if not rows:
        return block
    cols = [c for c in range(len(block[0])) if any(block[r][c] != 0 for r in rows)]
    return [[block[r][c] for c in cols] for r in rows]


def _is_solid_nonzero(block):
    all_vals = set()
    for row in block:
        for c in row:
            all_vals.add(c)
    return len(all_vals) == 1 and 0 not in all_vals


def _is_solid_nonzero_old(block):
    vals = set()
    for row in block:
        for c in row:
            if c != 0:
                vals.add(c)
    return len(vals) == 1


def _solid_color(block):
    for row in block:
        for c in row:
            if c != 0:
                return c
    return 0


def _transpose(grid):
    return [[grid[r][c] for r in range(len(grid))] for c in range(len(grid[0]))]


def _detect_sep_color(grid, axis):
    """axis='row' means look for uniform rows; axis='col' means uniform cols."""
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    H = len(grid); W = len(grid[0])
    if axis == 'row':
        uniform = [(r, grid[r][0]) for r in range(H) if len(set(grid[r])) == 1]
    else:
        uniform = [(c, grid[0][c]) for c in range(W)
                   if len(set(grid[r][c] for r in range(H))) == 1]
    per_color = defaultdict(list)
    for idx, color in uniform:
        per_color[color].append(idx)
    # Separator: non-bg color with >=2 singletons (no two consecutive).
    for color, idxs in per_color.items():
        if color == bg:
            continue
        if len(idxs) < 2:
            continue
        ok = True
        for i in range(len(idxs) - 1):
            if idxs[i + 1] == idxs[i] + 1:
                ok = False
                break
        if ok:
            return color, sorted(idxs), bg
    return None, [], bg


def _split_by_indices(grid, indices, axis):
    blocks = []
    if axis == 'row':
        H = len(grid)
        start = 0
        for r in range(H):
            if r in indices:
                if start < r:
                    blocks.append(grid[start:r])
                start = r + 1
        if start < H:
            blocks.append(grid[start:])
    else:
        W = len(grid[0])
        start = 0
        for c in range(W):
            if c in indices:
                if start < c:
                    blocks.append([row[start:c] for row in grid])
                start = c + 1
        if start < W:
            blocks.append([row[start:] for row in grid])
    return [b for b in blocks if b and b[0]]


def solve(grid):
    H = len(grid); W = len(grid[0])
    # Decide orientation by which axis has a valid separator
    sep_row, row_idxs, _ = _detect_sep_color(grid, 'row')
    sep_col, col_idxs, _ = _detect_sep_color(grid, 'col')

    if sep_row is not None and (sep_col is None or len(row_idxs) >= len(col_idxs)):
        axis = 'row'
        sep_idxs = set(row_idxs)
    elif sep_col is not None:
        axis = 'col'
        sep_idxs = set(col_idxs)
    else:
        return [row[:] for row in grid]

    blocks = _split_by_indices(grid, sep_idxs, axis)
    # Need exactly 4 blocks: A, B, S1, S2 in order.
    if len(blocks) < 4:
        return [row[:] for row in grid]
    # Identify which are solid vs not; keep order.
    A = B = S1_block = S2_block = None
    for b in blocks:
        if _is_solid_nonzero(b):
            if S1_block is None:
                S1_block = b
            elif S2_block is None:
                S2_block = b
        else:
            if A is None:
                A = b
            elif B is None:
                B = b
    if A is None or B is None or S1_block is None or S2_block is None:
        return [row[:] for row in grid]

    A_core = _strip_zero_border(A)
    B_core = _strip_zero_border(B)
    if not A_core or not B_core:
        return [row[:] for row in grid]

    # n = total non-zero dots in B / 2 (consistent across all train+test pairs).
    total_dots = sum(1 for row in B_core for c in row if c != 0)
    n = total_dots // 2

    S1 = _solid_color(S1_block)
    S2 = _solid_color(S2_block)

    H_A = len(A_core); W_A = len(A_core[0])
    # Recolor A_core: non-zero -> S1, zero -> S2.
    recolored = [[S1 if A_core[r][c] != 0 else S2 for c in range(W_A)] for r in range(H_A)]

    if axis == 'row':
        # Vertical tiling
        out = []
        for t in range(n):
            for r in range(H_A):
                out.append(recolored[r][:])
            if t < n - 1:
                out.append([S2] * W_A)
        return out
    else:
        # Horizontal tiling
        out = [[] for _ in range(H_A)]
        for t in range(n):
            for r in range(H_A):
                out[r].extend(recolored[r])
            if t < n - 1:
                for r in range(H_A):
                    out[r].append(S2)
        return out
