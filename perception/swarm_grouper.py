"""Post-process SWARM GROUPING (Option 1, 2026-09-20) -- collapse a cluster of
nearby same-class, same-side swarm units (a Skeleton Army's skeletons, a Goblin
Gang's goblins, Bats, Minions, Barbarians, ...) into ONE "group" entity with a
count and an enclosing box, instead of N separate pings.

This is deliberately a step AFTER detection/tracking, NOT a change to the detector
(see the design discussion 2026-09-20): the YOLO keeps detecting individuals (its
small-unit recall is already good, and individual positions stay available), and
this layer produces the blob view the DECISION layer actually wants ("a ~8-skeleton
swarm here", not eight coordinates). It's non-lossy -- callers keep the raw units
and can use whichever view they need -- needs no retrain, and is fully tunable.

Pure + framework-agnostic: operates on the unit dicts that thin_tracker (live) and
play_localizer.resolved_units (TV Royale) both emit (name/side/bbox, + optional
id/confidence/hp/level). Clustering is same-class + same-side + proximity
(union-find; adjacency scales with unit box size, so it works at any zoom). A
cluster of >= MIN_GROUP members becomes a group; smaller clusters and non-swarm
classes pass through UNCHANGED.
"""

import math

# Detector classes that deploy in numbers and clutter the board as a swarm. Kept an
# explicit allowlist so distinct same-class units that AREN'T a swarm (two Giants,
# two Musketeers) are never merged. Tunable.
SWARM_CLASSES = {
    "skeleton", "goblin", "spear_goblin", "bat", "minion", "barbarian",
    "elite_barbarian", "guard", "rascal_boy", "rascal_girl", "royal_recruit",
    "royal_hog",
}
MIN_GROUP = 2          # a cluster of this many same-class units becomes a group
GROUP_FACTOR = 1.5     # adjacency: two units link if centre-distance <= factor * mean box "size"
# (size = max(w,h)); swarm units are packed ~1 size apart, so 1.5 links a cluster
# while leaving units a couple of unit-widths apart as separate groups/individuals.


def _center(bbox):
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _size(bbox):
    return max(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _adjacent(a, b, factor):
    (ax, ay), (bx, by) = _center(a["bbox"]), _center(b["bbox"])
    d = math.hypot(ax - bx, ay - by)
    return d <= factor * (_size(a["bbox"]) + _size(b["bbox"])) / 2


def _union_bbox(units):
    xs1 = [u["bbox"][0] for u in units]
    ys1 = [u["bbox"][1] for u in units]
    xs2 = [u["bbox"][2] for u in units]
    ys2 = [u["bbox"][3] for u in units]
    return (min(xs1), min(ys1), max(xs2), max(ys2))


def _cluster(units, factor):
    """Union-find clusters of mutually-adjacent units (already same class+side)."""
    parent = list(range(len(units)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if _adjacent(units[i], units[j], factor):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(units)):
        groups.setdefault(find(i), []).append(units[i])
    return list(groups.values())


def _make_group(members):
    """Merge a cluster of same-class+side units into one group entity. Aggregates
    count, an enclosing bbox, a representative hp/level, and a stable-ish id (the
    smallest member id) so the blob keeps a steady identity frame-to-frame."""
    hps = [m["hp"] for m in members if isinstance(m.get("hp"), (int, float))]
    levels = [m["level"] for m in members if m.get("level") is not None]
    ids = [m["id"] for m in members if m.get("id") is not None]
    confs = [m["confidence"] for m in members if m.get("confidence") is not None]
    return {
        "name": members[0]["name"],
        "side": members[0].get("side", "unknown"),
        "bbox": _union_bbox(members),
        "count": len(members),
        "group": True,
        "member_ids": ids,
        "hp": (sum(hps) / len(hps)) if hps else None,
        "level": levels[0] if levels else None,
        "confidence": max(confs) if confs else None,
        "id": min(ids) if ids else None,
    }


def group_swarms(units, swarm_classes=SWARM_CLASSES, min_group=MIN_GROUP,
                 factor=GROUP_FACTOR):
    """Collapse swarm clusters into group entities; return the mixed list of groups +
    ungrouped units. Non-swarm classes and clusters below `min_group` pass through as
    their ORIGINAL dicts (untouched); a group is a NEW dict with count>=2 and
    group=True. Input order is not preserved (irrelevant to the consumers)."""
    swarm, out = [], []
    for u in units:
        (swarm if u.get("name") in swarm_classes else out).append(u)

    # bucket the swarm-eligible units by (class, side) so only like-with-like clusters
    buckets = {}
    for u in swarm:
        buckets.setdefault((u["name"], u.get("side", "unknown")), []).append(u)

    for members in buckets.values():
        for cluster in _cluster(members, factor):
            if len(cluster) >= min_group:
                out.append(_make_group(cluster))
            else:
                out.extend(cluster)          # too few to be a swarm -> keep individuals
    return out
