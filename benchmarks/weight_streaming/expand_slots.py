"""Give each arena slot class more slots: same capacities, tenants spread round-robin.

Each slot's tenants are reassigned, in first-use order (the plan's operation order),
to K copies of that slot, so a group no longer has to wait for the immediately
preceding tenant of its class to be freed. Slot capacities are unchanged; the
arena grows by (K - 1) x the expanded slots' sizes.

  python expand_slots.py PLAN.json OUT.json --copies 2 [--min-tenants 2]
"""
import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--min-tenants", type=int, default=2, help="only expand slots with at least this many tenants")
    a = ap.parse_args()

    plan = json.load(open(a.plan))
    slots = plan["summary"]["arena"]["slots"]
    reloads = [o for o in plan["io_operations"] if o.get("type") == "vram_prefetch_h2d"]
    order = []  # group ids in first-use order
    for o in reloads:
        if o["group_id"] not in order:
            order.append(o["group_id"])
    slot_of = {o["group_id"]: o["slot_id"] for o in reloads}
    tenants = {s["slot_id"]: [g for g in order if slot_of[g] == s["slot_id"]] for s in slots}

    next_id = max(s["slot_id"] for s in slots) + 1
    new_slot = dict(slot_of)
    for s in list(slots):
        sid, ts = s["slot_id"], tenants[s["slot_id"]]
        if len(ts) < a.min_tenants:
            continue
        copies = [sid] + list(range(next_id, next_id + a.copies - 1))
        next_id += a.copies - 1
        slots += [{"slot_id": c, "class_mb": s["class_mb"]} for c in copies[1:]]
        for i, g in enumerate(ts):
            new_slot[g] = copies[i % len(copies)]
    for o in plan["io_operations"]:
        if o.get("group_id") in new_slot and "slot_id" in o:
            o["slot_id"] = new_slot[o["group_id"]]
    plan["summary"]["n_slots"] = len(slots)
    plan["summary"]["expand_slots"] = {"copies": a.copies, "slots": len(slots),
                                       "arena_mb": round(sum(s["class_mb"] for s in slots), 1)}
    json.dump(plan, open(a.out, "w"))
    print(json.dumps(plan["summary"]["expand_slots"]))


if __name__ == "__main__":
    main()
