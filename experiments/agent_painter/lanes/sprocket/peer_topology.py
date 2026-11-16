#!/usr/bin/env python3
"""Pinned read-only cross-lane *proposal* graph-identity screen.

Not a shared TRAIN registry, held-out disjointness certificate, native admission
or permission to launch training. These adjacency witnesses are transcribed
from VOID proposal source 54161e2217f709ffb8b23634f21efa351411395f,
blob 446f0ed7a6f7f16f8de9a6bfd97ebbde3a425b16.
Peer files are never modified; future peer revisions require renewed review.
"""
from __future__ import annotations
from collections import Counter

PEER_SOURCE_SHA = "54161e2217f709ffb8b23634f21efa351411395f"
PEER_BLOB_SHA = "446f0ed7a6f7f16f8de9a6bfd97ebbde3a425b16"
PEER_GRAPH_EDGES = {
    "void-pipe-P01": (6, ((0,1),(1,2),(2,3),(1,4),(2,5))),
    "void-pipe-P04": (6, ((0,1),(1,2),(2,3),(3,0),(0,4),(2,5))),
    "void-pipe-P05": (8, ((0,1),(1,2),(2,3),(3,0),(2,4),(4,5),(5,6),(6,2),(0,7))),
    "void-pipe-P06": (9, ((0,1),(1,2),(3,4),(4,5),(6,7),(7,8),
                          (0,3),(3,6),(1,4),(4,7),(2,5),(5,8))),
    "void-pipe-P07": (6, ((0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,4))),
}

def adjacency(node_count: int, edges) -> tuple[frozenset[int], ...]:
    if type(node_count) is not int or not 1 <= node_count <= 24:
        raise ValueError("invalid node count")
    adj = [set() for _ in range(node_count)]
    seen = set()
    for edge in edges:
        if len(edge) != 2 or any(type(x) is not int or not 0 <= x < node_count for x in edge):
            raise ValueError("invalid edge")
        a,b = edge
        e = frozenset((a,b))
        if a == b or e in seen:
            raise ValueError("self-loop or duplicate edge")
        seen.add(e)
        adj[a].add(b)
        adj[b].add(a)
    if any(not n for n in adj):
        raise ValueError("isolated graph node")
    return tuple(frozenset(n) for n in adj)

def isomorphic(na: int, ea, nb: int, eb) -> bool:
    """Exact adjacency equivalence; deliberately ignores XY, seed, ID and width."""
    if na != nb:
        return False
    a,b = adjacency(na,ea),adjacency(nb,eb)
    da,db = tuple(map(len,a)),tuple(map(len,b))
    if Counter(da) != Counter(db):
        return False
    fa = tuple((da[i],tuple(sorted(da[j] for j in a[i]))) for i in range(na))
    fb = tuple((db[i],tuple(sorted(db[j] for j in b[i]))) for i in range(nb))
    if Counter(fa) != Counter(fb):
        return False
    options = {i:tuple(j for j in range(nb) if fa[i]==fb[j]) for i in range(na)}
    order = sorted(range(na),key=lambda i:(len(options[i]),-da[i],i))
    mapping,used = {},set()
    def recurse(depth):
        if depth == na:
            return True
        i=order[depth]
        for j in options[i]:
            if j in used or any((k in a[i]) != (v in b[j]) for k,v in mapping.items()):
                continue
            mapping[i]=j
            used.add(j)
            if recurse(depth+1):
                return True
            used.remove(j)
            del mapping[i]
        return False
    return recurse(0)

def peer_matches(nodes,edges) -> list[str]:
    return [k for k,(n,e) in PEER_GRAPH_EDGES.items()
            if isomorphic(len(nodes),edges,n,e)]
