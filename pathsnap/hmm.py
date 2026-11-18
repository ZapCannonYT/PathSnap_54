"""HMM map-matching decoded with Viterbi.

Hidden state at observation t : one candidate arc of GPS point t (with the
                                projection of the point on that arc).
Emission  log P(z_t | s)      : tau * log(classifier probability of s).
Transition log P(s_t | s_t-1) : -|route_dist - gc_dist| / beta, where gc_dist is the
                                straight-line distance between the two GPS points and
                                route_dist is the driving distance along the network
                                between the two projections (bounded Dijkstra).
                                Unreachable pairs get -inf; a U-turn onto the reverse
                                twin of the same road gets an extra UTURN_M penalty.
Chain breaks                  : if the time gap exceeds gap_s, or no transition is
                                reachable, the current segment is closed and a new
                                Viterbi segment starts (decoded independently).
Points without candidates     : skipped by the decoder, filled with the previous arc.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

UTURN_M = 200.0
MAX_SEARCH_M = 3000.0


@dataclass
class HMMParams:
    beta: float = 20.0      # metres; scale of the route-vs-straight-line mismatch
    tau: float = 1.0        # weight of the classifier emission
    gap_s: float = 120.0    # break the chain after a longer GPS gap
    slack_m: float = 150.0  # extra search distance beyond 2 x gc_dist


class _Graph:
    """Adjacency lists for a bounded Dijkstra that only touches nearby nodes."""

    def __init__(self, net):
        order = np.argsort(net.arcs[:, 0], kind="stable")
        self.dst = net.arcs[order, 1]
        self.w = net.length[order]
        self.ptr = np.searchsorted(net.arcs[order, 0], np.arange(net.n_nodes + 1))

    def dijkstra(self, src: int, targets: set, limit: float) -> dict:
        dist = {src: 0.0}
        heap = [(0.0, src)]
        found = {}
        remaining = set(targets)
        while heap and remaining:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            if u in remaining:
                found[u] = d
                remaining.discard(u)
            for k in range(self.ptr[u], self.ptr[u + 1]):
                v, nd = int(self.dst[k]), d + float(self.w[k])
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return found

    def path(self, src: int, dst: int, limit: float, arc_index: dict) -> list:
        """Arc ids of the shortest path src -> dst (empty if not found)."""
        dist, prev = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, np.inf):
                continue
            for k in range(self.ptr[u], self.ptr[u + 1]):
                v, nd = int(self.dst[k]), d + float(self.w[k])
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        if dst not in prev and dst != src:
            return []
        nodes = [dst]
        while nodes[-1] != src:
            nodes.append(prev[nodes[-1]])
        nodes.reverse()
        return [arc_index[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]


def _transition(net, g, a_arc, a_frac, b_arc, b_frac, gc, p: HMMParams):
    """Matrix of log transition probabilities (len(a) x len(b))."""
    la, lb = net.length[a_arc], net.length[b_arc]
    route = np.full((len(a_arc), len(b_arc)), np.inf)

    same = a_arc[:, None] == b_arc[None, :]
    along = np.abs(b_frac[None, :] * lb[None, :] - a_frac[:, None] * la[:, None])
    route[same] = along[same]

    twin = net.twin[a_arc][:, None] == b_arc[None, :]
    route[twin] = (np.abs((1 - b_frac[None, :]) * lb[None, :] - a_frac[:, None] * la[:, None]) + UTURN_M)[twin]

    limit = min(2.0 * gc + p.slack_m, MAX_SEARCH_M)
    ends = net.arcs[a_arc, 1]
    starts = net.arcs[b_arc, 0]
    need = ~(same | twin)
    for src in np.unique(ends):
        rows = np.flatnonzero(ends == src)
        cols = np.flatnonzero(need[rows].any(axis=0))
        if len(cols) == 0:
            continue
        found = g.dijkstra(int(src), set(int(s) for s in starts[cols]), limit)
        for c in cols:
            sp = found.get(int(starts[c]))
            if sp is None:
                continue
            r = (1 - a_frac[rows]) * la[rows] + sp + b_frac[c] * lb[c]
            route[rows, c] = np.where(need[rows, c], r, route[rows, c])
    return -np.abs(route - gc) / p.beta


def viterbi_decode(net, xy: np.ndarray, t: np.ndarray, cands, proba: np.ndarray,
                   params: HMMParams | None = None, graph: _Graph | None = None):
    """Decode one trip.

    cands: DataFrame for this trip with point_idx, arc_id, proj_frac (any order).
    proba: classifier probability per row of cands.
    Returns (arc per point with -1 where undecided, number of chain breaks).
    """
    p = params or HMMParams()
    g = graph or _Graph(net)
    n = len(xy)
    order = np.lexsort((-proba, cands["point_idx"].to_numpy()))
    pi = cands["point_idx"].to_numpy()[order]
    arc = cands["arc_id"].to_numpy()[order]
    frac = cands["proj_frac"].to_numpy()[order]
    emis = p.tau * np.log(np.clip(proba[order], 1e-6, 1.0))
    bounds = np.searchsorted(pi, np.arange(n + 1))

    out = np.full(n, -1, dtype=np.int64)
    breaks = 0
    seg = []                 # list of (point, arcs, backpointer)
    delta = None
    prev_pt = None

    def close_segment():
        if not seg:
            return
        k = int(np.argmax(delta))
        for (pt, arcs, bp) in reversed(seg):
            out[pt] = arcs[k]
            if bp is not None:
                k = int(bp[k])
        seg.clear()

    for i in range(n):
        lo, hi = bounds[i], bounds[i + 1]
        if lo == hi:
            continue
        a_i, f_i, e_i = arc[lo:hi], frac[lo:hi], emis[lo:hi]
        if delta is None or t[i] - t[prev_pt] > p.gap_s:
            if delta is not None:
                breaks += 1
            close_segment()
            delta = e_i.copy()
            seg.append((i, a_i, None))
        else:
            plo, phi = bounds[prev_pt], bounds[prev_pt + 1]
            gc = float(np.hypot(*(xy[i] - xy[prev_pt])))
            trans = _transition(net, g, arc[plo:phi], frac[plo:phi], a_i, f_i, gc, p)
            score = delta[:, None] + trans
            bp = np.argmax(score, axis=0)
            best = score[bp, np.arange(len(a_i))]
            if not np.isfinite(best).any():
                breaks += 1
                close_segment()
                delta = e_i.copy()
                seg.append((i, a_i, None))
            else:
                delta = best + e_i
                seg.append((i, a_i, bp))
        prev_pt = i
    close_segment()
    return out, breaks


def fill_missing(arcs: np.ndarray) -> np.ndarray:
    """Give points without a decoded arc the previous (or next) decoded arc."""
    a = arcs.copy()
    idx = np.where(a >= 0, np.arange(len(a)), -1)
    idx = np.maximum.accumulate(idx)
    first = np.flatnonzero(a >= 0)
    if len(first) == 0:
        return a
    idx[idx < 0] = first[0]
    return a[idx]


def reconstruct_route(net, point_arcs: np.ndarray, graph: _Graph | None = None) -> list:
    """Ordered arc list: consecutive duplicates removed, small disconnections bridged
    with the shortest network path (left as a jump when no path exists)."""
    g = graph or _Graph(net)
    arc_index = {(int(a), int(b)): i for i, (a, b) in enumerate(net.arcs)}
    seq = [int(a) for a in point_arcs if a >= 0]
    route = []
    for a in seq:
        if route and route[-1] == a:
            continue
        if route and net.arcs[route[-1], 1] != net.arcs[a, 0]:
            route.extend(g.path(int(net.arcs[route[-1], 1]), int(net.arcs[a, 0]), MAX_SEARCH_M, arc_index))
        route.append(a)
    return route
