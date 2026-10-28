"""Overnight batch: keep the asset lane fed with unique props from a catalog for N hours, one 3D asset per prompt.

  python tools/overnight.py --hours 8                 # run (resumes from tools/overnight_state.json if present)
  python tools/overnight.py --list                    # show the catalog and what is already done
  python tools/overnight.py --dry-run --hours 8       # print the plan, submit nothing
  python tools/overnight.py --catalog my_catalog.json # your own catalog (same shape as COZY below)

The script submits one-shot jobs (prompt -> reference image -> Pixal3D -> Blender) through the local API and keeps at
most --depth jobs queued at a time, so stopping it (Ctrl+C) leaves only a couple of queued jobs behind (it cancels
those). Progress is written to tools/overnight_state.json and a summary to tools/overnight_report.md, so a restart
continues where it left off and never repeats an asset. Prompts follow the no-text rule (signs are blank).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = os.environ.get("STUDIO_URL", "http://127.0.0.1:8090")
TOKEN = os.environ.get("STUDIO_TOKEN", "")
HERE = Path(__file__).resolve().parent

# The cozy look, applied on top of the generic stylized preset (same fields as the portal's style editor).
COZY_STYLE = {
    "label": "Cozy",
    "style_clause": "cozy hand-painted game asset with a warm storybook charm, soft rounded shapes, gentle proportions, "
                    "inviting warm colours, subtle painted texture and light wear, friendly and comfortable, "
                    "readable silhouette, no tiny protrusions",
    "background": "plain flat warm light grey studio background",
    "negative_extra": "gloomy, dark, rusty, grimy, horror, sharp spikes, photorealistic, clutter",
}

# name, prompt, materials, palette, height (m), triangle budget
COZY = [
    ("Rocking chair", "wooden rocking chair with a knitted cushion and a folded blanket on the seat", "warm oak wood, wool knit", ["honey oak", "cream", "sage green"], 1.1, 12000),
    ("Reading armchair", "overstuffed reading armchair with a rolled back and a small throw pillow", "velvet upholstery, wood legs", ["dusty rose", "walnut", "cream"], 1.0, 14000),
    ("Wood stove", "small cast-iron wood stove with a kettle on top and a stovepipe", "cast iron, copper kettle", ["charcoal", "copper", "brick red"], 1.2, 14000),
    ("Stone fireplace", "cottage stone fireplace with a wooden mantel and logs stacked in its side nook", "fieldstone, oak mantel", ["warm grey stone", "oak", "ember orange"], 1.8, 20000),
    ("Bookshelf", "short wooden bookshelf packed with books of different sizes and a potted plant on top", "pine wood, paper", ["pine", "teal", "mustard", "cream"], 1.4, 16000),
    ("Kitchen table", "round farmhouse kitchen table with a checked tablecloth and a bowl of apples", "painted wood, cotton cloth", ["cream", "red check", "green apple"], 0.8, 12000),
    ("Kitchen chair", "simple painted farmhouse kitchen chair with a woven rush seat", "painted wood, rush", ["sage green", "straw"], 0.9, 8000),
    ("Bread oven", "domed clay bread oven with a wooden peel leaning against it", "clay, brick base, wood", ["terracotta", "sand", "oak"], 1.5, 14000),
    ("Bakery display", "bakery counter display with loaves, buns and croissants on wooden trays", "wood, pastry", ["golden brown", "cream", "oak"], 1.0, 16000),
    ("Tea set", "porcelain teapot with two cups on a small round tray", "glazed porcelain, wood tray", ["cream", "dusty blue", "walnut"], 0.25, 8000),
    ("Coffee grinder", "vintage hand-cranked coffee grinder with a small drawer", "walnut wood, brass", ["walnut", "brass"], 0.3, 8000),
    ("Copper pot rack", "hanging pot rack with copper pots and pans on hooks", "copper, iron rack", ["copper", "dark iron"], 0.6, 12000),
    ("Pantry shelf", "open pantry shelf with jars of jam, flour sacks and baskets", "pine shelf, glass jars, burlap", ["pine", "berry red", "amber", "burlap"], 1.6, 16000),
    ("Kitchen sink", "farmhouse kitchen sink with a wooden counter and a hanging towel", "ceramic, oak counter, brass tap", ["white ceramic", "oak", "brass"], 0.9, 12000),
    ("Cottage bed", "cozy single bed with a patchwork quilt and two plump pillows", "wood frame, patchwork fabric", ["pine", "rose", "sky blue", "cream"], 0.9, 14000),
    ("Bedside lamp", "small bedside lamp with a fabric shade on a turned wooden base", "wood, linen shade", ["oak", "cream"], 0.4, 6000),
    ("Wardrobe", "painted cottage wardrobe with two doors and a small floral carving", "painted wood", ["mint", "cream", "brass handles"], 1.9, 12000),
    ("Dresser", "small chest of drawers with round wooden knobs and a lace runner on top", "painted wood, lace", ["powder blue", "cream", "oak"], 1.0, 10000),
    ("Vanity mirror", "oval standing vanity mirror in a carved wooden frame", "carved wood, mirror glass", ["walnut", "gold accent"], 1.5, 8000),
    ("Bathtub", "clawfoot bathtub with a wooden tray and a folded towel", "enamelled iron, wood", ["white enamel", "copper feet", "oak"], 0.7, 10000),
    ("Garden bench", "wooden garden bench with a curved back and a small climbing rose wound around one armrest", "weathered wood, rose vine", ["driftwood grey", "leaf green", "pink"], 0.9, 12000),
    ("Wishing well", "small stone wishing well with a shingled roof and a bucket on a rope", "fieldstone, wood shingles, rope", ["warm stone", "moss green", "oak"], 2.0, 16000),
    ("Birdhouse", "birdhouse on a wooden post with a tiny round door and a shingled roof", "painted wood", ["cream", "teal roof", "oak post"], 1.8, 6000),
    ("Mailbox", "rounded countryside mailbox on a wooden post with a little flag", "painted metal, wood post", ["red", "cream", "oak"], 1.2, 6000),
    ("Wheelbarrow", "wooden wheelbarrow full of pumpkins", "wood, iron wheel, pumpkins", ["oak", "orange", "iron"], 0.7, 10000),
    ("Watering can", "rounded tin watering can with a long spout and a rose head", "painted tin", ["sage green", "brass"], 0.35, 6000),
    ("Garden gnome", "cheerful garden gnome with a pointed hat holding a tiny lantern", "painted ceramic", ["red hat", "blue coat", "beige"], 0.45, 8000),
    ("Flower cart", "small wooden flower cart with buckets of tulips and daisies", "wood, tin buckets, flowers", ["cream", "yellow", "pink", "green"], 1.3, 16000),
    ("Beehive", "traditional skep beehive on a short wooden stand", "straw, wood", ["straw", "oak", "lavender"], 0.9, 10000),
    ("Rain barrel", "wooden rain barrel with a small tap and a tin cup hanging on it", "oak staves, iron hoops", ["oak", "dark iron"], 1.0, 8000),
    ("Vegetable planter", "raised wooden planter box with rows of lettuce and carrots", "pine, soil, vegetables", ["pine", "leaf green", "carrot orange"], 0.6, 12000),
    ("Greenhouse", "small cottage greenhouse with a peaked glass roof and potted plants inside", "painted wood frame, glass", ["cream", "leaf green", "terracotta"], 2.6, 20000),
    ("Chicken coop", "small chicken coop with a ramp and a rounded nesting box", "painted wood, wire", ["barn red", "cream", "straw"], 1.6, 14000),
    ("Hay cart", "small two-wheeled hay cart piled with hay bales", "wood, hay", ["oak", "straw yellow"], 1.3, 12000),
    ("Apple tree", "small round apple tree with red apples and a little wooden ladder", "bark, leaves, apples", ["bark brown", "leaf green", "apple red"], 3.0, 20000),
    ("Market stall", "small covered market stall with a striped awning and crates of vegetables", "wood, canvas, produce", ["oak", "red stripe", "cream", "green"], 2.4, 20000),
    ("Fruit crate", "wooden fruit crate full of oranges", "pine crate, oranges", ["pine", "orange"], 0.3, 6000),
    ("Milk churn", "tall metal milk churn with a lid and two handles", "galvanised steel", ["silver grey", "brass"], 0.75, 6000),
    ("Cheese wheel stand", "wooden stand stacked with cheese wheels of different sizes", "wood, wax cheese", ["oak", "cream", "orange wax"], 0.8, 10000),
    ("Honey jar", "glass honey jar with a wooden dipper and a cloth-topped lid", "glass, honey, wood", ["amber", "cream", "oak"], 0.15, 5000),
    ("Lantern", "round iron lantern with a small candle inside and a carrying ring", "wrought iron, glass", ["black iron", "warm glow"], 0.35, 6000),
    ("Candle cluster", "cluster of three pillar candles on a wooden dish", "wax, wood", ["cream", "walnut"], 0.25, 5000),
    ("Oil lamp", "brass oil lamp with a glass chimney", "brass, glass", ["brass", "clear"], 0.4, 6000),
    ("Street lamp", "short cast-iron village street lamp with a rounded glass head", "cast iron, glass", ["dark green", "warm glow"], 2.8, 8000),
    ("Fairy light string", "wooden post with a string of round paper lanterns", "wood, paper", ["oak", "cream", "peach"], 2.2, 8000),
    ("Teddy bear", "plush teddy bear sitting with a little scarf", "plush fabric", ["caramel", "red scarf"], 0.4, 8000),
    ("Toy train", "wooden toy train with three small carriages", "painted wood", ["red", "blue", "yellow"], 0.15, 8000),
    ("Rocking horse", "wooden rocking horse with a painted saddle", "painted wood, rope mane", ["cream", "red saddle", "oak"], 0.9, 10000),
    ("Doll house", "small wooden doll house with an open front and tiny furniture", "painted wood", ["pink roof", "cream", "oak"], 0.7, 16000),
    ("Kite", "diamond kite with a long ribbon tail leaning against a wooden post", "paper, wood", ["sky blue", "yellow", "red"], 1.5, 6000),
    ("Snowman", "friendly snowman with a knitted hat, scarf and a carrot nose", "snow, wool", ["white", "red", "green", "orange"], 1.4, 10000),
    ("Sled", "wooden sled with red runners and a rope", "wood, painted metal", ["oak", "red"], 0.4, 6000),
    ("Ice skates", "pair of leather ice skates hanging on a wooden peg", "leather, steel blades", ["tan leather", "steel"], 0.4, 6000),
    ("Hot cocoa mug", "big ceramic mug of hot cocoa with marshmallows on a small saucer", "glazed ceramic", ["dusty blue", "cocoa brown", "white"], 0.15, 5000),
    ("Pumpkin pile", "pile of pumpkins of different sizes with a few autumn leaves", "pumpkin skin, leaves", ["orange", "cream", "green", "red leaf"], 0.6, 10000),
    ("Scarecrow", "friendly scarecrow with a straw hat and a patched shirt on a wooden pole", "straw, cloth, wood", ["straw", "blue shirt", "brown"], 2.0, 12000),
    ("Log pile", "small wooden firewood shelter filled with neatly stacked logs", "birch logs, wood", ["birch white", "oak"], 1.4, 12000),
    ("Axe stump", "wood-chopping stump with an axe stuck in the top", "wood, steel axe", ["oak", "steel"], 0.6, 6000),
    ("Water pump", "old-fashioned cast-iron hand water pump with a wooden bucket", "cast iron, wood", ["dark green iron", "oak"], 1.3, 8000),
    ("Village fountain", "small round stone village fountain with a central spout", "carved stone, water", ["warm stone", "blue water"], 1.5, 14000),
    ("Wooden bridge", "small arched wooden footbridge with rope handrails", "wood planks, rope", ["oak", "rope beige"], 1.2, 12000),
    ("Signpost", "wooden village signpost with three blank arrow boards pointing different ways", "wood", ["oak", "cream boards"], 2.2, 6000),
    ("Picket gate", "white picket garden gate with a small arch of roses", "painted wood, roses", ["white", "leaf green", "pink"], 1.6, 10000),
    ("Stone wall", "short mossy stone wall section with a wooden gate", "fieldstone, moss, wood", ["grey stone", "moss", "oak"], 1.2, 12000),
    ("Cottage door", "rounded cottage front door in a stone frame with a small lantern and a flower pot", "painted wood, stone, iron", ["teal door", "warm stone", "terracotta"], 2.2, 12000),
    ("Bay window", "cottage bay window with flower boxes and lace curtains", "painted wood, glass, flowers", ["cream", "pink", "leaf green"], 1.8, 12000),
    ("Chimney", "brick cottage chimney with a little smoke cap", "brick, clay pot", ["brick red", "terracotta"], 1.6, 8000),
    ("Thatched hut", "tiny round thatched storage hut with a wooden door", "thatch, plaster, wood", ["straw", "cream plaster", "oak"], 3.0, 16000),
    ("Garden shed", "small wooden garden shed with a tool rack and a window", "painted wood, shingles", ["sage green", "cream trim", "oak"], 2.6, 16000),
    ("Treehouse", "small treehouse platform with a rope ladder in a rounded tree", "wood, rope, foliage", ["oak", "leaf green", "rope"], 4.0, 20000),
    ("Windmill", "small stone windmill with wooden sails and a round door", "stone, wood sails, canvas", ["warm stone", "oak", "cream"], 6.0, 20000),
    ("Dovecote", "round wooden dovecote on a post with little arched openings", "painted wood, shingles", ["cream", "grey shingles", "oak"], 2.4, 10000),
    ("Wagon", "covered wooden wagon with a rounded canvas top", "wood, canvas, iron wheels", ["oak", "cream canvas", "red wheels"], 2.4, 16000),
    ("Rowboat", "small wooden rowboat with two oars and a coiled rope", "painted wood, rope", ["sky blue", "cream", "oak"], 0.8, 12000),
    ("Fishing pier", "short wooden fishing pier section with a lantern post and a bucket", "wood, rope, iron", ["driftwood", "rope", "dark iron"], 1.4, 12000),
    ("Bicycle", "vintage bicycle with a wicker basket of flowers", "painted steel, wicker", ["mint", "wicker", "pink"], 1.1, 12000),
    ("Wicker basket", "round wicker picnic basket with a checked cloth and a baguette", "wicker, cloth, bread", ["wicker", "red check", "golden"], 0.35, 8000),
    ("Picnic table", "wooden picnic table with attached benches and a lemonade jug", "wood, glass", ["oak", "lemon yellow"], 0.8, 12000),
    ("Hammock", "rope hammock on a wooden hammock stand with a folded blanket", "rope, wood, wool", ["rope beige", "oak", "rust red"], 1.4, 10000),
    ("Campfire", "small campfire ring of stones with logs and a hanging cooking pot on a tripod", "stone, wood, iron", ["grey stone", "oak", "ember orange"], 1.0, 12000),
    ("Tent", "small canvas camping tent with a rolled-up door flap", "canvas, wood poles, rope", ["cream canvas", "oak", "rope"], 1.6, 10000),
    ("Sleeping bag", "rolled sleeping bag with a leather strap", "quilted fabric, leather", ["forest green", "tan"], 0.3, 5000),
    ("Backpack", "canvas hiking backpack with leather straps and a rolled blanket on top", "canvas, leather, wool", ["olive", "tan", "red"], 0.6, 10000),
    ("Book stack", "stack of old books with a pair of round spectacles on top", "leather-bound paper, brass", ["burgundy", "navy", "cream", "brass"], 0.25, 6000),
    ("Writing desk", "small slanted writing desk with an inkwell and a quill", "walnut wood, glass, feather", ["walnut", "ink blue", "cream"], 1.0, 12000),
    ("Globe", "small antique globe on a wooden stand", "painted globe, wood", ["parchment", "ocean blue", "walnut"], 0.5, 8000),
    ("Grandfather clock", "tall grandfather clock with a round blank face and a pendulum", "walnut wood, brass", ["walnut", "brass", "cream face"], 2.0, 12000),
    ("Sewing machine", "vintage hand-crank sewing machine on a wooden base with a spool of thread", "painted iron, wood", ["black", "gold accent", "oak"], 0.35, 10000),
    ("Yarn basket", "basket overflowing with balls of yarn and knitting needles", "wicker, wool", ["wicker", "mustard", "teal", "rose"], 0.35, 8000),
    ("Spinning wheel", "wooden spinning wheel with a bundle of wool", "turned wood, wool", ["oak", "cream wool"], 1.0, 12000),
    ("Loom", "small tabletop weaving loom with a half-finished striped cloth", "wood, yarn", ["pine", "rust", "teal", "cream"], 0.6, 10000),
    ("Pottery wheel", "pottery wheel with an unfinished clay bowl and a few finished pots", "wood, clay", ["oak", "terracotta"], 0.8, 10000),
    ("Vase set", "three rounded ceramic vases of different heights with dried flowers", "glazed ceramic, dried flowers", ["sage", "cream", "terracotta", "wheat"], 0.5, 8000),
    ("Wall clock", "round wooden wall clock with a blank face and a small pendulum", "wood, brass", ["oak", "cream", "brass"], 0.5, 6000),
    ("Coat rack", "wooden standing coat rack with a hat, scarf and umbrella", "wood, felt, fabric", ["walnut", "red scarf", "grey hat"], 1.8, 10000),
    ("Umbrella stand", "ceramic umbrella stand with two umbrellas and a walking cane", "ceramic, fabric, wood", ["blue ceramic", "yellow", "oak"], 0.8, 8000),
    ("Welcome mat", "coiled rope doormat with a pair of boots standing on it", "rope, leather", ["rope beige", "brown leather"], 0.3, 6000),
    ("Boots", "pair of muddy leather boots with wool socks folded over", "leather, wool", ["brown", "cream"], 0.35, 6000),
    ("Cat bed", "round cat bed with a sleeping cat curled inside", "fabric, fur", ["cream", "ginger"], 0.3, 8000),
    ("Dog house", "small wooden dog house with a rounded door and a food bowl", "painted wood, shingles", ["red", "cream", "steel bowl"], 1.0, 10000),
    ("Bird feeder", "hanging wooden bird feeder with seeds and two small birds", "wood, seeds, feathers", ["pine", "seed brown", "blue bird"], 0.4, 8000),
    ("Fish tank", "small round goldfish bowl on a wooden stand with a plant inside", "glass, water, wood", ["clear", "orange", "green", "oak"], 0.4, 6000),
    ("Radio", "vintage wooden tabletop radio with round dials", "wood veneer, fabric grille", ["walnut", "cream", "brass"], 0.35, 8000),
    ("Gramophone", "gramophone with a brass horn on a wooden box", "brass, wood", ["brass", "walnut"], 0.7, 10000),
    ("Piano", "small upright cottage piano with a candle holder and open sheet stand", "wood, ivory keys", ["walnut", "cream", "black"], 1.3, 14000),
    ("Guitar", "acoustic guitar leaning on a wooden stool", "wood, strings", ["honey wood", "oak stool"], 1.0, 10000),
    ("Chess table", "small chess table with a wooden board and carved pieces", "wood, carved pieces", ["walnut", "maple", "cream"], 0.75, 12000),
    ("Board games shelf", "shelf of stacked board game boxes with a dice cup", "cardboard, wood", ["red", "blue", "mustard", "oak"], 1.0, 10000),
    ("Bakery cake", "layered cake on a glass stand with strawberries", "sponge, cream, glass", ["cream", "strawberry red", "pink"], 0.3, 8000),
    ("Pie", "lattice-topped berry pie in a ceramic dish", "pastry, ceramic", ["golden", "berry purple", "cream"], 0.12, 6000),
    ("Soup pot", "round enamel soup pot with a ladle on a wooden trivet", "enamelled steel, wood", ["cream enamel", "red rim", "oak"], 0.3, 6000),
    ("Bread basket", "cloth-lined basket of bread rolls", "wicker, cloth, bread", ["wicker", "cream", "golden"], 0.2, 6000),
    ("Herb rack", "wooden rack of small potted herbs", "wood, terracotta, herbs", ["oak", "terracotta", "green"], 0.5, 8000),
    ("Spice shelf", "small wall shelf of rounded spice jars with cork lids", "wood, glass, cork", ["oak", "paprika red", "turmeric", "cork"], 0.4, 8000),
    ("Ice cream cart", "small ice cream cart with a striped umbrella and a bell", "painted wood, canvas, brass", ["cream", "pink stripe", "mint"], 2.0, 16000),
    ("Lemonade stand", "small lemonade stand with a blank sign board and a pitcher", "painted wood, glass", ["yellow", "white", "lemon"], 1.6, 12000),
    ("Postbox", "round pillar postbox with a domed top", "cast iron", ["red", "black"], 1.6, 6000),
    ("Bench with cat", "park bench with a sleeping cat on one end", "wood, cast iron, fur", ["oak", "dark green iron", "grey cat"], 0.9, 12000),
    ("Wagon wheel", "old wooden wagon wheel leaning against a barrel", "wood, iron rim", ["oak", "dark iron"], 1.0, 8000),
    ("Stool", "three-legged milking stool with a folded cloth", "wood, cloth", ["oak", "red check"], 0.4, 5000),
    ("Ladder", "short wooden ladder with a paint can hanging from a rung", "wood, tin", ["oak", "sky blue"], 1.8, 6000),
    ("Trellis", "wooden garden trellis covered in climbing beans and flowers", "wood, plants", ["oak", "leaf green", "white flower"], 1.8, 12000),
    ("Sunflower pot", "large terracotta pot with three tall sunflowers", "terracotta, plants", ["terracotta", "yellow", "green"], 1.6, 10000),
    ("Mushroom cluster", "cluster of round red-capped mushrooms on a mossy log", "mushroom, moss, wood", ["red", "cream", "moss green", "brown"], 0.4, 8000),
    ("Acorn lamp", "whimsical lamp shaped like a large acorn on a wooden base", "wood, glass", ["oak", "warm glow"], 0.5, 8000),
    ("Hot air balloon", "small tethered hot air balloon with a wicker basket, striped envelope", "fabric, wicker, rope", ["cream", "peach", "teal", "wicker"], 4.0, 14000),
]


def api(method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method, headers={"Content-Type": "application/json"})
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"done": {}, "submitted": {}, "failed": {}}


def save_state(path: Path, st: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def submit(item, seed: int, style: str, quality: str, candidates: int) -> str:
    name, prompt, materials, palette, height, tris = item
    body = {
        "prompt": prompt, "style": style, "custom_style": COZY_STYLE, "quality": quality, "seed": seed,
        "materials": materials, "palette": palette, "height_m": height, "target_triangles": tris,
        "texture_size": 2048 if tris >= 10000 else 1024, "reference_candidates": candidates, "title": f"Cozy: {name}",
    }
    return api("POST", "/v1/jobs", body)["job_id"]


def queue_depth() -> int:
    q = api("GET", "/v1/queue")
    items = q if isinstance(q, list) else q.get("items", [])
    return sum(1 for j in items if j.get("kind") == "asset" and j.get("status") in ("queued", "running"))


def report(state: dict, catalog, path: Path, started: float):
    by_name = {c[0]: c for c in catalog}
    rows = []
    for name, rec in state["done"].items():
        rows.append((rec.get("finished", 0), name, rec["job_id"], "completed", rec.get("elapsed_s"), rec.get("triangles")))
    for name, rec in state["failed"].items():
        rows.append((rec.get("finished", 0), name, rec["job_id"], "failed: " + (rec.get("error") or "")[:80], None, None))
    rows.sort()
    lines = [f"# Overnight cozy batch — started {time.strftime('%Y-%m-%d %H:%M', time.localtime(started))}", "",
             f"{len(state['done'])} completed, {len(state['failed'])} failed, {len(catalog) - len(state['done'])} left in the catalog", "",
             "| # | Asset | Job | Result | Time | Triangles |", "|---|---|---|---|---|---|"]
    for i, (_, name, jid, res, el, tris) in enumerate(rows, 1):
        lines.append(f"| {i} | {name} | `{jid}` | {res} | {round(el) if el else ''} s | {tris or ''} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--depth", type=int, default=2, help="asset jobs to keep queued/running")
    ap.add_argument("--style", default="stylized_generic", help="base preset the cozy overrides apply to")
    ap.add_argument("--quality", default="balanced", choices=["balanced", "quality"])
    ap.add_argument("--candidates", type=int, default=2, help="reference images per asset (auto-selected)")
    ap.add_argument("--seed", type=int, default=20260913, help="catalog shuffle seed")
    ap.add_argument("--catalog", help="JSON list of [name, prompt, materials, palette, height_m, triangles]")
    ap.add_argument("--state", default=str(HERE / "overnight_state.json"))
    ap.add_argument("--report", default=str(HERE / "overnight_report.md"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="print progress, in-flight jobs and the GPU state, then exit")
    ap.add_argument("--resubmit-failed", action="store_true", help="forget failed items so they are generated again")
    a = ap.parse_args()

    catalog = [tuple(x) for x in json.loads(Path(a.catalog).read_text(encoding="utf-8"))] if a.catalog else list(COZY)
    names = [c[0] for c in catalog]
    assert len(names) == len(set(names)), "catalog names must be unique"
    scene_words = (" between ", " beside ", " next to ", " under a ", " around the base")
    for c in catalog:  # the pipeline wants exactly one object; warn about prompts that read like a small scene
        if any(w in " " + c[1] + " " for w in scene_words):
            print(f"[overnight] note: '{c[0]}' reads like a scene ({c[1][:60]}...); reconstruction prefers one object", flush=True)
    state_path, report_path = Path(a.state), Path(a.report)
    state = load_state(state_path)
    if a.resubmit_failed and state["failed"]:
        print(f"[overnight] forgetting {len(state['failed'])} failed item(s): {', '.join(state['failed'])}", flush=True)
        state["failed"] = {}
        save_state(state_path, state)
    if a.status:
        print(f"done {len(state['done'])}  failed {len(state['failed'])}  in flight {len(state['submitted'])}  "
              f"catalog {len(catalog)}")
        for name, rec in state["submitted"].items():
            try:
                j = api("GET", f"/v1/jobs/{rec['job_id']}")
                print(f"  {name:22s} {rec['job_id']} {j['status']} {j.get('stage')} {round((j.get('progress') or 0) * 100)}%")
            except Exception as e:  # noqa: BLE001
                print(f"  {name:22s} {rec['job_id']} (api: {e})")
        for name, rec in list(state["failed"].items())[-5:]:
            print(f"  failed {name}: {(rec.get('error') or '')[:100]}")
        try:
            import subprocess
            print("  gpu:", subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                                          capture_output=True, text=True, timeout=10).stdout.strip())
        except Exception:  # noqa: BLE001
            pass
        return
    order = list(catalog)
    random.Random(a.seed).shuffle(order)
    todo = [c for c in order if c[0] not in state["done"] and c[0] not in state["submitted"]]
    if a.list:
        for c in order:
            mark = "done" if c[0] in state["done"] else ("failed" if c[0] in state["failed"] else ("queued" if c[0] in state["submitted"] else ""))
            print(f"{mark:7s} {c[0]:22s} {c[1][:70]}")
        print(f"\n{len(catalog)} in catalog, {len(state['done'])} done, {len(todo)} to do")
        return
    deadline = time.time() + a.hours * 3600
    started = time.time()
    print(f"[overnight] {len(todo)} assets to do, {a.hours:g} h budget, depth {a.depth}, {a.candidates} candidate(s) each; "
          f"state {state_path.name}", flush=True)
    if a.dry_run:
        for c in todo[:20]:
            print(f"  would submit: {c[0]} — {c[1]}")
        return
    try:
        api("GET", "/health")
    except Exception as e:  # noqa: BLE001
        print(f"[overnight] API not reachable at {API}: {e}", flush=True)
        sys.exit(1)

    stop = {"now": False}

    def _sig(*_):
        stop["now"] = True
        print("\n[overnight] stopping: no more submissions; cancelling jobs still queued", flush=True)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    est_job_s = 330.0  # rolling estimate of a full asset job (image + 3D + optimise)
    last_report = 0.0
    while not stop["now"]:
        # 1. poll submitted jobs
        for name, rec in list(state["submitted"].items()):
            try:
                j = api("GET", f"/v1/jobs/{rec['job_id']}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    state["submitted"].pop(name)
                continue
            except Exception:  # noqa: BLE001
                continue
            if j["status"] == "completed":
                el = (j.get("finished_at") or time.time()) - (j.get("started_at") or rec["submitted"])
                tris = (j.get("summary") or {}).get("triangles")
                state["done"][name] = {"job_id": rec["job_id"], "finished": time.time(), "elapsed_s": el, "triangles": tris}
                state["submitted"].pop(name)
                est_job_s = 0.7 * est_job_s + 0.3 * el
                print(f"[overnight] done {name} in {el:.0f}s ({rec['job_id']}) — {len(state['done'])} completed, "
                      f"~{est_job_s:.0f}s/asset, {(deadline - time.time()) / 3600:.1f} h left", flush=True)
            elif j["status"] == "failed" and not rec.get("retried"):
                # one retry resumes from the failed stage (completed stages are reused): ~30 s if Blender/validate failed
                err = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else str(j.get("error"))
                try:
                    api("POST", f"/v1/jobs/{rec['job_id']}/retry", {})
                    rec["retried"] = time.time()
                    print(f"[overnight] retrying {name} once after: {(err or '')[:120]}", flush=True)
                except Exception as e:  # noqa: BLE001
                    rec["retried"] = time.time()
                    print(f"[overnight] retry request failed for {name}: {e}", flush=True)
            elif j["status"] in ("failed", "cancelled"):
                err = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else str(j.get("error"))
                state["failed"][name] = {"job_id": rec["job_id"], "finished": time.time(), "error": err}
                state["submitted"].pop(name)
                print(f"[overnight] FAILED {name} ({rec['job_id']}): {err}", flush=True)
            save_state(state_path, state)
        # 2. top up the queue while there is time for the job to finish before the deadline
        try:
            depth = queue_depth()
        except Exception:  # noqa: BLE001
            depth = a.depth
        remaining = deadline - time.time()
        while todo and depth < a.depth and remaining > est_job_s * 1.2 and not stop["now"]:
            item = todo.pop(0)
            seed = random.Random(f"{a.seed}:{item[0]}").randint(0, 2**31 - 1)
            try:
                jid = submit(item, seed, a.style, a.quality, a.candidates)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")[:300]
                print(f"[overnight] submit rejected for {item[0]}: {e.code} {body}", flush=True)
                state["failed"][item[0]] = {"job_id": "", "finished": time.time(), "error": f"submit {e.code}: {body}"}
                save_state(state_path, state)
                continue
            state["submitted"][item[0]] = {"job_id": jid, "submitted": time.time()}
            save_state(state_path, state)
            depth += 1
            print(f"[overnight] submitted {item[0]} -> {jid} ({len(todo)} left)", flush=True)
        if time.time() - last_report > 300:
            report(state, catalog, report_path, started)
            last_report = time.time()
        if not todo and not state["submitted"]:
            print("[overnight] catalog exhausted", flush=True)
            break
        if remaining <= 0 and not state["submitted"]:
            print("[overnight] time budget used up", flush=True)
            break
        time.sleep(20)
    if stop["now"]:
        for name, rec in list(state["submitted"].items()):
            try:
                j = api("GET", f"/v1/jobs/{rec['job_id']}")
                if j["status"] == "queued":
                    api("POST", f"/v1/jobs/{rec['job_id']}/cancel")
                    state["submitted"].pop(name)
                    print(f"[overnight] cancelled queued {name}", flush=True)
            except Exception:  # noqa: BLE001
                pass
        save_state(state_path, state)
    report(state, catalog, report_path, started)
    print(f"[overnight] {len(state['done'])} completed, {len(state['failed'])} failed. Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
