"""Independent literal-composition checks for prospective SMOL source-v2 drawings.

These checks are intentionally narrower than the compiler's structural validator.
They test whether the *authored* programs satisfy their own language descriptions.
They are not native receipts, global cross-lane novelty approval, or TRAIN admission.
"""
from collections import Counter


def require(ok, message):
    if not ok:
        raise ValueError("composition witness: " + message)


def _interval_overlap(a, aw, b, bw):
    return max(0, min(a + aw, b + bw) - max(a, b))


def inspect(item):
    family, p = item["family"], item["program"]
    require(p["family"] == family, "family mismatch")
    components = p["components"]
    if family == "greebles":
        require(len(components) == 3, "expected exactly three cassettes")
        panels = sorted(components, key=lambda c: c["box"][0])
        boxes = [c["box"] for c in panels]
        require(len({(b[2], b[3]) for b in boxes}) == 3, "cassette dimensions repeat")
        require(len({c["detail"] for c in panels}) == 3, "all cassette details must differ")
        require(all(c["fasteners"] is True for c in panels), "all cassettes require fasteners")
        heights = [b[1] + b[3] / 2 for b in boxes]
        require(heights[0] > heights[1] > heights[2], "not an ascending side-elevation progression")
        require(all(heights[i] - heights[i+1] >= 10 for i in (0, 1)),
                "diagonal ascent too shallow")
        gaps = [boxes[i+1][0] - boxes[i][0] - boxes[i][2] for i in (0, 1)]
        require(gaps[0] >= 1 and gaps[1] >= gaps[0] + 12,
                "missing intermediate cassette is not represented by a larger gap")
        require(p["requirements"]["negative_space_min_percent"] >= 75,
                "negative-space target lost")
        return {"family": family, "panel_boxes": boxes, "center_y": heights,
                "horizontal_void_gaps": gaps, "different_detail_kinds": 3,
                "missing_interval": "gap_2_greater_than_gap_1_by_at_least_12"}

    if family == "pipework":
        require(len(components) == 1 and components[0]["kind"] == "pipe_network",
                "expected one explicit network")
        graph = components[0]
        nodes, edges = graph["nodes"], graph["edges"]
        require(len(nodes) == 7 and len(edges) == 6, "not a seven-node tree")
        adjacency = [set() for _ in nodes]
        for u, v in edges:
            require(u != v and v not in adjacency[u], "duplicate/self edge")
            adjacency[u].add(v)
            adjacency[v].add(u)
        degree = [len(n) for n in adjacency]
        require(Counter(degree) == Counter({1: 5, 3: 1, 4: 1}),
                "not five endpoints with degree-three and degree-four junctions")
        lower = degree.index(3)
        upper = degree.index(4)
        lx, ly = nodes[lower]
        ux, uy = nodes[upper]
        require(upper in adjacency[lower] and lx == ux and uy < ly - 10,
                "second junction not vertically above first")
        lower_leaves = [nodes[i] for i in adjacency[lower] if i != upper]
        upper_leaves = [nodes[i] for i in adjacency[upper] if i != lower]
        require(len(lower_leaves) == 2 and all(y == ly for x, y in lower_leaves)
                and min(x for x, y in lower_leaves) < lx < max(x for x, y in lower_leaves),
                "first tee lacks opposing horizontal ends")
        require(len(upper_leaves) == 3 and
                sum(x < ux and y == uy for x, y in upper_leaves) == 1 and
                sum(x > ux and y == uy for x, y in upper_leaves) == 1 and
                sum(x == ux and y < uy for x, y in upper_leaves) == 1,
                "upper junction does not split into left/right/up destinations")
        require(graph["diameter"] == 3 and p["requirements"] == {"connected": True, "cycles": 0},
                "uniform 3-cell connected acyclic contract changed")
        return {"family": family, "degree_multiset": sorted(degree), "endpoints": 5,
                "junctions": 2, "cycles": 0, "first_junction": [lx, ly],
                "upper_junction": [ux, uy], "consistent_diameter": 3}

    require(family == "architecture", "unsupported family")
    require(len(components) == 4 and all(c["kind"] == "room" for c in components),
            "expected one hall and three upper rooms")
    hall = max(components, key=lambda c: c["box"][2])
    bays = sorted((c for c in components if c is not hall), key=lambda c: c["box"][0])
    hx, hy, hw, hh = hall["box"]
    require(hw >= 70 and hw >= 2 * max(c["box"][2] for c in bays),
            "dominant horizontal hall missing")
    require(len({(c["box"][2], c["box"][3]) for c in bays}) == 3,
            "unequal alcove dimensions lost")
    require(all(b["box"][1] + b["box"][3] == hy for b in bays),
            "upper alcoves do not touch hall roof line")
    require(all(bays[i]["box"][0] + bays[i]["box"][2] < bays[i+1]["box"][0]
                for i in (0, 1)), "upper alcoves overlap")
    tops = sorted((o for o in hall["openings"] if o["side"] == "top"),
                  key=lambda o: o["offset"])
    require(len(tops) == 3, "hall must have three distinct upper roof doors")
    matching = []
    for bay, opening in zip(bays, tops):
        bottoms = [o for o in bay["openings"] if o["side"] == "bottom"]
        require(len(bottoms) == 1, "each upper bay needs one explicit bottom door")
        b = bottoms[0]
        overlap = _interval_overlap(hx + opening["offset"], opening["size"],
                                    bay["box"][0] + b["offset"], b["size"])
        require(overlap >= 5, "bay-to-hall upper/lower door alignment is too narrow")
        matching.append(overlap)
    route = p["requirements"]["route"]
    clear = p["requirements"]["clearance"]
    require(clear == 5 and len(route) == 2 and route[0][1] == route[1][1],
            "five-cell straight geometric corridor lost")
    half = clear // 2
    require(route[0][0] < route[1][0] and
            all(hx + hall["thickness"] + half <= x < hx + hw - hall["thickness"] - half
                for x, y in route) and
            all(hy + hall["thickness"] + half <= y < hy + hh - hall["thickness"] - half
                for x, y in route),
            "five-cell route not contained within hall interior")
    return {"family": family, "hall_box": hall["box"],
            "alcove_boxes": [c["box"] for c in bays],
            "aligned_door_spans": matching, "clearance": 5,
            "route": route, "claim": "geometric clearance only; not gravity traversal"}
