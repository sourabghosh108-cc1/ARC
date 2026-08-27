"""Task 3e6067c3 â€” Indicator->Projection (key sequence walks a graph of frames).

STRUCTURE
  Grid of bordered rectangular "frames", each with a uniform border color and
  a single non-border interior marker color (which may appear in only one or
  a few interior cells). Frames are arranged on a regular row/col grid with
  background-color gaps. One row (near bottom) or column (near right) is the
  "key row": alternating-bg sequence of colors, terminating with consecutive bg.

INDICATOR
  Key sequence [C0, C1, ..., Cn-1]. Each Ci identifies a frame visited by a
  walk on the frame grid; consecutive frames must be spatially adjacent.

PROJECTION
  For each consecutive pair (Ci, Ci+1) in the walk: fill the gap between the
  two adjacent frames with color Ci, restricted to rows (or cols) where the
  marker color of frame_i appears.

ALGORITHM
  1. Detect background color (mode of grid).
  2. Detect key line (row or column) with alternating bg/non-bg pattern,
     terminated by consecutive bg. Extract sequence of colors.
  3. Find all frames: bordered rectangles (non-bg connected components of size
     >= 3x3) with a single non-border, non-bg interior marker color.
  4. Walk: starting from a frame matching C0, at each step move to an adjacent
     frame matching C_{i+1}. Adjacency = bbox row-overlap (horiz) or col-overlap
     (vert). Prefer the closest adjacent frame.
  5. Fill the gap between consecutive frames in the walk with color Ci, at
     marker_rows (horiz) or marker_cols (vert).
"""
from collections import Counter


def _bg_color(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


def _parse_alt(line, bg):
    n = len(line)
    if n < 4:
        return None
    colors = []
    i = 0
    while i < n:
        if line[i] != bg:
            return None
        if i + 1 >= n:
            break
        nxt = line[i + 1]
        if nxt == bg:
            break
        colors.append(nxt)
        i += 2
    j = i
    while j < n:
        if line[j] != bg:
            return None
        j += 1
    return colors if colors else None


def _detect_key_line(grid, bg):
    H, W = len(grid), len(grid[0])
    for r in range(H):
        colors = _parse_alt(grid[r], bg)
        if colors and len(colors) >= 3:
            return ('row', r, colors)
    for c in range(W):
        col = [grid[r][c] for r in range(H)]
        colors = _parse_alt(col, bg)
        if colors and len(colors) >= 3:
            return ('col', c, colors)
    return None


def _find_frames(grid, bg, key_kind, key_idx):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    frames = []
    for r in range(H):
        if key_kind == 'row' and r == key_idx:
            continue
        for c in range(W):
            if key_kind == 'col' and c == key_idx:
                continue
            if visited[r][c] or grid[r][c] == bg:
                continue
            stack = [(r, c)]
            cells = []
            while stack:
                cr, cc = stack.pop()
                if not (0 <= cr < H and 0 <= cc < W):
                    continue
                if visited[cr][cc] or grid[cr][cc] == bg:
                    continue
                if key_kind == 'row' and cr == key_idx:
                    continue
                if key_kind == 'col' and cc == key_idx:
                    continue
                visited[cr][cc] = True
                cells.append((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cr + dr, cc + dc))
            if not cells:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = grid[r0][c0]
            marker_cells = []
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    v = grid[rr][cc]
                    if v != border and v != bg:
                        marker_cells.append((rr, cc, v))
            if not marker_cells:
                continue
            colors_in = set(v for _, _, v in marker_cells)
            if len(colors_in) != 1:
                continue
            ic = next(iter(colors_in))
            mrows = sorted(set(rr for rr, _, _ in marker_cells))
            mcols = sorted(set(cc for _, cc, _ in marker_cells))
            frames.append({
                'color': ic,
                'border': border,
                'bbox': (r0, r1, c0, c1),
                'marker_rows': mrows,
                'marker_cols': mcols,
            })
    return frames


def _adjacency(f1, f2):
    r0a, r1a, c0a, c1a = f1['bbox']
    r0b, r1b, c0b, c1b = f2['bbox']
    row_overlap = not (r1a < r0b or r1b < r0a)
    col_overlap = not (c1a < c0b or c1b < c0a)
    if row_overlap and not col_overlap:
        return 'horiz'
    if col_overlap and not row_overlap:
        return 'vert'
    return None


def _gap_distance(f1, f2):
    kind = _adjacency(f1, f2)
    if kind is None:
        return 10**9
    r0a, r1a, c0a, c1a = f1['bbox']
    r0b, r1b, c0b, c1b = f2['bbox']
    if kind == 'horiz':
        if c1a < c0b:
            return c0b - c1a - 1
        return c0a - c1b - 1
    else:
        if r1a < r0b:
            return r0b - r1a - 1
        return r0a - r1b - 1


def _walk(frames, sequence):
    by_color = {}
    for f in frames:
        by_color.setdefault(f['color'], []).append(f)
    if sequence[0] not in by_color:
        return None
    # Try each possible start; return longest valid walk. Prefer one of length n.
    best = None
    for start in by_color[sequence[0]]:
        path = [start]
        for ci in sequence[1:]:
            curr = path[-1]
            cands = []
            for f in by_color.get(ci, []):
                if f is curr:
                    continue
                d = _gap_distance(curr, f)
                if d < 10**9:
                    cands.append((d, f))
            if not cands:
                break
            cands.sort(key=lambda x: x[0])
            path.append(cands[0][1])
        if best is None or len(path) > len(best):
            best = path
        if len(best) == len(sequence):
            return best
    return best


def solve(grid):
    grid = [list(row) for row in grid]
    H, W = len(grid), len(grid[0])
    bg = _bg_color(grid)
    key = _detect_key_line(grid, bg)
    if key is None:
        return grid
    key_kind, key_idx, sequence = key
    frames = _find_frames(grid, bg, key_kind, key_idx)
    if not frames:
        return grid

    path = _walk(frames, sequence)
    if path is None or len(path) < 2:
        return grid

    for i in range(len(path) - 1):
        f1 = path[i]
        f2 = path[i + 1]
        kind = _adjacency(f1, f2)
        if kind is None:
            continue
        r0a, r1a, c0a, c1a = f1['bbox']
        r0b, r1b, c0b, c1b = f2['bbox']
        ci = sequence[i]
        if kind == 'horiz':
            if c1a < c0b:
                gap_cols = range(c1a + 1, c0b)
            else:
                gap_cols = range(c1b + 1, c0a)
            for r in f1['marker_rows']:
                for c in gap_cols:
                    if grid[r][c] == bg:
                        grid[r][c] = ci
        else:
            if r1a < r0b:
                gap_rows = range(r1a + 1, r0b)
            else:
                gap_rows = range(r1b + 1, r0a)
            for c in f1['marker_cols']:
                for r in gap_rows:
                    if grid[r][c] == bg:
                        grid[r][c] = ci
    return grid
