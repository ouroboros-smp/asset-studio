"""Headless Blender (bpy 4.5 LTS) post-processing for a Pixal3D master GLB.

Steps (all conservative; the master file is never modified):
  1. import master GLB, join into one mesh, apply transforms
  2. normalise orientation/scale (requested physical size), origin at bottom-centre
  3. remove loose verts/edges, degenerate faces, obvious debris (tiny far-from-budget components only)
  4. optimized static asset at the requested triangle budget
       - light reduction: decimate with UV seams delimited, master textures reused (resized to texture_size)
       - heavy reduction: decimate, new UVs (Smart UV Project), rebake base colour / metallic / roughness /
         alpha and a tangent-space normal map from the master (Cycles, CPU)
  5. optional LODs (further decimation, textures reused) and a convex-hull collision mesh
  6. neutral-lit preview renders (Cycles CPU) + contact sheet
  7. GLB export (glTF Y-up, embedded PNG textures) and statistics

Everything requested is reported in output.json; a map is listed only if it was generated or baked.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Vector

WARN: list[str] = []
STATS: dict = {}
T: dict = {}


def log(msg):
    print(f"[blender] {msg}", flush=True)


def tick(name, t0):
    T[name] = round(time.time() - t0, 2)


# --------------------------------------------------------------------------------------------- scene helpers

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0


def select_only(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def import_master(path: str):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path, import_shading="NORMALS")
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("master GLB contains no mesh")
    select_only(meshes)
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        select_only(meshes)
        bpy.ops.object.join()
        WARN.append(f"master contained {len(meshes)} mesh objects; joined into one")
    obj = bpy.context.view_layer.objects.active
    for o in new:
        if o.type != "MESH" and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    obj.name = "master"
    return obj


def tri_count(obj) -> int:
    me = obj.data
    n = len(me.polygons)
    if n == 0:
        return 0
    lt = np.empty(n, dtype=np.int32)
    me.polygons.foreach_get("loop_total", lt)
    return int(np.maximum(lt - 2, 0).sum())


def bounds(obj):
    co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return co.min(0), co.max(0)


def duplicate(obj, name):
    new = obj.copy()
    new.data = obj.data.copy()
    new.name = name
    new.data.name = name
    bpy.context.scene.collection.objects.link(new)
    return new


def apply_modifiers(obj):
    select_only([obj])
    bpy.ops.object.convert(target="MESH")


# --------------------------------------------------------------------------------------------- normalisation

def normalise(obj, opt: dict) -> dict:
    mn, mx = bounds(obj)
    dims = mx - mn
    # Blender is Z-up here (glTF importer converts). X width, Y depth, Z height.
    scale = 1.0
    basis = None
    if opt.get("height_m"):
        scale, basis = opt["height_m"] / max(dims[2], 1e-9), "height_m"
    elif opt.get("width_m"):
        scale, basis = opt["width_m"] / max(dims[0], 1e-9), "width_m"
    elif opt.get("depth_m"):
        scale, basis = opt["depth_m"] / max(dims[1], 1e-9), "depth_m"
    else:
        WARN.append(f"no physical dimensions requested; master kept at generated scale (height {dims[2]:.3f} m)")
    for k, i in (("width_m", 0), ("depth_m", 1), ("height_m", 2)):
        if opt.get(k) and basis and k != basis:
            got = dims[i] * scale
            if abs(got - opt[k]) > 0.05 * opt[k]:
                WARN.append(f"{k}={opt[k]} not applied (uniform scale from {basis} gives {got:.3f} m); non-uniform scaling is not applied")
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    center = np.array([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]], dtype=np.float32)
    co = (co - center) * scale
    me.vertices.foreach_set("co", co.reshape(-1))
    me.update()
    obj.location = (0, 0, 0)
    mn, mx = bounds(obj)
    return {"scale_factor": float(scale), "scale_basis": basis, "dimensions_m": {"x": float(mx[0] - mn[0]), "y": float(mx[1] - mn[1]), "z": float(mx[2] - mn[2])},
            "origin": "bottom_center"}


# --------------------------------------------------------------------------------------------- cleanup

def _components(me) -> tuple[np.ndarray, int]:
    """Connected-component label per polygon via vectorised label propagation on shared vertices (numpy only)."""
    me.calc_loop_triangles()
    n_tri = len(me.loop_triangles)
    tri = np.empty(n_tri * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    poly_of_tri = np.empty(n_tri, dtype=np.int64)
    me.loop_triangles.foreach_get("polygon_index", poly_of_tri)
    nv = len(me.vertices)
    # merge vertices that share a position (UV-seam splits must not break connectivity)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    span = float(np.ptp(co, axis=0).max()) or 1.0
    q = np.round(co / span * 1e5).astype(np.int64)
    _, pos_id = np.unique(q, axis=0, return_inverse=True)
    tri = pos_id.reshape(-1)[tri]
    label = np.arange(int(pos_id.max()) + 1, dtype=np.int64)
    for _ in range(2000):
        prev = label
        fmin = np.minimum(np.minimum(label[tri[:, 0]], label[tri[:, 1]]), label[tri[:, 2]])
        label = label.copy()
        np.minimum.at(label, tri[:, 0], fmin)
        np.minimum.at(label, tri[:, 1], fmin)
        np.minimum.at(label, tri[:, 2], fmin)
        # pointer jumping speeds convergence
        label = label[label]
        if np.array_equal(prev, label):
            break
    tri_label = label[tri[:, 0]]
    poly_label = np.full(len(me.polygons), -1, dtype=np.int64)
    poly_label[poly_of_tri] = tri_label
    _, inv = np.unique(poly_label, return_inverse=True)
    return inv, int(inv.max()) + 1 if len(inv) else 0


def renormalise(obj, opt: dict):
    """After decimation/proxy the bounds drift by a few mm: re-apply the exact requested height and bottom-centre origin."""
    mn, mx = bounds(obj)
    h = float(mx[2] - mn[2]) or 1.0
    scale = (opt["height_m"] / h) if opt.get("height_m") else 1.0
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    center = np.array([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]], dtype=np.float32)
    me.vertices.foreach_set("co", ((co - center) * scale).reshape(-1))
    me.update()


def cleanup(obj) -> dict:
    """Conservative cleanup with C-level operators: loose verts/edges, degenerate faces, obvious debris only."""
    me = obj.data
    before_v, before_f = len(me.vertices), len(me.polygons)
    mn, mx = bounds(obj)
    diag = float(np.linalg.norm(mx - mn)) or 1.0
    select_only([obj])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_degenerate(threshold=diag * 1e-7)
    bpy.ops.object.mode_set(mode="OBJECT")
    after_loose_v, after_loose_f = len(me.vertices), len(me.polygons)
    comp, n_comp = _components(me)
    removed_debris = 0
    if n_comp > 1:
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        pv = np.empty(len(me.polygons), dtype=np.int64)
        me.polygons.foreach_get("loop_start", pv)  # any vertex of the polygon is enough for extent estimation
        loops_v = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", loops_v)
        first_v = loops_v[pv]
        counts = np.bincount(comp, minlength=n_comp)
        total = len(me.polygons)
        kill = np.zeros(n_comp, dtype=bool)
        small = np.where(counts < max(4, 0.0005 * total))[0]
        for c in small:
            pts = co[first_v[comp == c]]
            ext = float(np.linalg.norm(pts.max(0) - pts.min(0)))
            if ext < 0.005 * diag:
                kill[c] = True
        if kill.any():
            sel = kill[comp]
            me.vertices.foreach_set("select", np.zeros(len(me.vertices), dtype=bool))
            me.edges.foreach_set("select", np.zeros(len(me.edges), dtype=bool))
            me.polygons.foreach_set("select", sel)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_mode(type="FACE")
            bpy.ops.mesh.delete(type="FACE")
            bpy.ops.object.mode_set(mode="OBJECT")
            removed_debris = int(kill.sum())
    rep = {"vertices_before": before_v, "faces_before": before_f, "loose_or_degenerate_faces_removed": before_f - after_loose_f,
           "loose_vertices_removed": before_v - after_loose_v, "components": int(n_comp), "debris_components_removed": removed_debris,
           "vertices_after": len(me.vertices), "faces_after": len(me.polygons)}
    if n_comp > 1:
        rep["note"] = "disconnected parts preserved (only tiny debris removed)"
    return rep


# --------------------------------------------------------------------------------------------- materials / images

def _trace_image(socket):
    """Walk upstream from a node socket until an Image Texture node is found."""
    stack = [socket]
    seen = set()
    while stack:
        s = stack.pop()
        for link in s.links:
            n = link.from_node
            if n.bl_idname == "ShaderNodeTexImage":
                return n.image
            if n.name in seen:
                continue
            seen.add(n.name)
            stack.extend(i for i in n.inputs)
    return None


def glb_material_info(path: str) -> dict:
    """Read alphaMode/doubleSided of the first material straight from the GLB JSON chunk (importer-independent)."""
    import struct
    with open(path, "rb") as f:
        magic, _ver, _len = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            return {}
        clen, ctype = struct.unpack("<II", f.read(8))
        if ctype != 0x4E4F534A:
            return {}
        j = json.loads(f.read(clen).decode("utf-8"))
    mats = j.get("materials") or []
    if not mats:
        return {}
    return {"alpha_mode": mats[0].get("alphaMode", "OPAQUE"), "double_sided": bool(mats[0].get("doubleSided", False)),
            "alpha_cutoff": mats[0].get("alphaCutoff", 0.5)}


def master_maps(obj) -> dict:
    mat = obj.active_material
    if mat is None or not mat.use_nodes:
        raise RuntimeError("master has no node material")
    bsdf = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        raise RuntimeError("master material has no Principled BSDF")
    maps = {"base_color": _trace_image(bsdf.inputs["Base Color"]), "metallic": _trace_image(bsdf.inputs["Metallic"]),
            "roughness": _trace_image(bsdf.inputs["Roughness"]), "alpha": _trace_image(bsdf.inputs["Alpha"]),
            "normal": _trace_image(bsdf.inputs["Normal"])}
    return {"material": mat, "bsdf": bsdf, "images": maps, "alpha_mode": mat.blend_method if hasattr(mat, "blend_method") else None}


def image_to_np(img) -> np.ndarray:
    w, h = img.size
    arr = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(arr)
    return arr.reshape(h, w, 4)


def np_to_image(name, arr, is_data, path: Path):
    h, w = arr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=False)
    if is_data:
        img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).reshape(-1))
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    img.filepath = str(path)
    img.source = "FILE"
    img.reload()
    if is_data:
        img.colorspace_settings.name = "Non-Color"
    return img


def resized_copy(img, size, name, path: Path, is_data):
    arr = image_to_np(img)
    if max(arr.shape[:2]) != size:
        tmp = img.copy()
        tmp.scale(size, size)
        arr = image_to_np(tmp)
        bpy.data.images.remove(tmp)
    return np_to_image(name, arr, is_data, path)


def build_pbr_material(name, base_img, mr_img, normal_img=None, alpha_mode="OPAQUE"):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    tb = nt.nodes.new("ShaderNodeTexImage")
    tb.image = base_img
    tb.image.colorspace_settings.name = "sRGB"
    nt.links.new(tb.outputs["Color"], bsdf.inputs["Base Color"])
    if alpha_mode != "OPAQUE":
        nt.links.new(tb.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.blend_method = "BLEND" if alpha_mode == "BLEND" else "CLIP"
    tm = nt.nodes.new("ShaderNodeTexImage")
    tm.image = mr_img
    tm.image.colorspace_settings.name = "Non-Color"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tm.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    if normal_img is not None:
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = normal_img
        tn.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.space = "TANGENT"
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# --------------------------------------------------------------------------------------------- baking

def _emission_material(name, img, channel: str | None, is_data: bool):
    """Temporary material that emits one channel of an image (used to bake exact PBR channels without lighting)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    if img is None:  # constant white emission (hit mask)
        em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        return mat
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.image.colorspace_settings.name = "Non-Color" if is_data else "sRGB"
    if channel is None:
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    elif channel == "A":
        nt.links.new(tex.outputs["Alpha"], em.inputs["Color"])
    else:
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(tex.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs[{"R": "Red", "G": "Green", "B": "Blue"}[channel]], em.inputs["Color"])
    return mat


def _bake_target(obj, size, name, is_data):
    img = bpy.data.images.new(name, size, size, alpha=True, float_buffer=False)
    if is_data:
        img.colorspace_settings.name = "Non-Color"
    mat = bpy.data.materials.new(name + "_bake")
    mat.use_nodes = True
    nt = mat.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.nodes.active = tex
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return img, mat


def fill_missed(arr: np.ndarray, hit: np.ndarray, iters: int = 8) -> np.ndarray:
    """Replace texels whose bake ray missed the master (hit=False) with the mean of their hit neighbours, growing
    outwards by one texel per iteration (also a small margin around islands). `arr` may stack several maps along the
    last axis so all of them are filled in one pass. Stops early once no fillable texel is left."""
    out = arr.copy()
    ok = hit.copy()
    H, W = ok.shape
    for _ in range(iters):
        # 4-neighbour dilation with slicing (no np.roll copies of the full stack)
        nb_ok = np.zeros((H, W), dtype=np.uint8)
        acc = np.zeros_like(out)
        for src_sl, dst_sl in (((slice(1, None), slice(None)), (slice(0, -1), slice(None))),
                               ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
                               ((slice(None), slice(1, None)), (slice(None), slice(0, -1))),
                               ((slice(None), slice(0, -1)), (slice(None), slice(1, None)))):
            m = ok[src_sl]
            nb_ok[dst_sl] += m
            acc[dst_sl] += out[src_sl] * m[..., None]
        fillable = (~ok) & (nb_ok > 0)
        if not fillable.any():
            break
        out[fillable] = acc[fillable] / nb_ok[fillable][:, None]
        ok |= fillable
    return out


def bake_from_master(master, target, size: int, maps: dict, out_dir: Path, diag: float, prefix: str = "asset",
                     cage: float | None = None) -> dict:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.render.bake.use_selected_to_active = True
    cage = cage if cage is not None else diag * 0.01
    sc.render.bake.cage_extrusion = max(cage, 1e-4)
    sc.render.bake.max_ray_distance = max(diag * 0.06, cage * 6, 1e-3)
    sc.render.bake.margin = max(4, size // 256)
    sc.render.bake.use_clear = True
    results = {}
    tex_dir = out_dir / "textures"
    tex_dir.mkdir(exist_ok=True)
    base = maps["images"]["base_color"]
    mr = maps["images"]["metallic"] or maps["images"]["roughness"]
    channel_jobs = [("base_color", base, None, False, "EMIT")]
    if mr is not None:
        channel_jobs += [("mr", mr, None, True, "EMIT")]  # one bake: G=roughness, B=metallic (glTF packing)
    if base is not None and maps["alpha_mode"] not in (None, "OPAQUE"):
        channel_jobs.append(("alpha", base, "A", True, "EMIT"))
    baked = {}
    orig_master_mats = [m for m in master.data.materials]
    for key, img, ch, is_data, btype in channel_jobs:
        if img is None:
            continue
        t0 = time.time()
        em = _emission_material(f"tmp_emit_{key}", img, ch, is_data)
        master.data.materials.clear()
        master.data.materials.append(em)
        tgt, tmat = _bake_target(target, size, f"bake_{key}", is_data)
        select_only([master, target])
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.bake(type=btype, use_selected_to_active=True, margin=sc.render.bake.margin)
        baked[key] = image_to_np(tgt)
        bpy.data.images.remove(tgt)
        bpy.data.materials.remove(tmat)
        bpy.data.materials.remove(em)
        results[f"bake_{key}_s"] = round(time.time() - t0, 2)
    # hit mask: constant white emission -> black texels are rays that missed the master
    em = _emission_material("tmp_emit_hit", None, None, True)
    master.data.materials.clear()
    master.data.materials.append(em)
    tgt, tmat = _bake_target(target, size, "bake_hit", True)
    select_only([master, target])
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.bake(type="EMIT", use_selected_to_active=True, margin=0)
    hit = image_to_np(tgt)[..., :3].mean(-1) > 0.5
    bpy.data.images.remove(tgt)
    bpy.data.materials.remove(tmat)
    bpy.data.materials.remove(em)
    results["bake_missed_texels"] = int((~hit).sum())
    # normal map from the master geometry
    t0 = time.time()
    master.data.materials.clear()
    for m in orig_master_mats:
        master.data.materials.append(m)
    tgt, tmat = _bake_target(target, size, "bake_normal", True)
    select_only([master, target])
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_selected_to_active=True, margin=sc.render.bake.margin)
    baked["normal"] = image_to_np(tgt)
    bpy.data.images.remove(tgt)
    bpy.data.materials.remove(tmat)
    results["bake_normal_s"] = round(time.time() - t0, 2)
    if "mr" in baked:
        baked["roughness"] = baked["mr"][..., 1:2]
        baked["metallic"] = baked["mr"][..., 2:3]
        del baked["mr"]
    # fill ray misses (and grow island margins) from nearest hit texels: one pass over all maps stacked
    names = list(baked)
    widths = [baked[k].shape[-1] for k in names]
    stacked = fill_missed(np.concatenate([baked[k] for k in names], axis=-1), hit)
    off = 0
    for k, w in zip(names, widths):
        baked[k] = stacked[..., off:off + w]
        off += w
    # compose glTF images
    b = baked["base_color"]
    if "alpha" in baked:
        b = b.copy()
        b[..., 3] = baked["alpha"][..., 0]
    else:
        b = b.copy()
        b[..., 3] = 1.0
    base_img = np_to_image(f"{prefix}_baseColor", b, False, tex_dir / f"{prefix}_baseColor.png")
    mr_arr = np.zeros_like(b)
    mr_arr[..., 3] = 1.0
    if "roughness" in baked:
        mr_arr[..., 1] = baked["roughness"][..., 0]
        mr_arr[..., 2] = baked["metallic"][..., 0]
    else:
        mr_arr[..., 1] = 0.5
    mr_img = np_to_image(f"{prefix}_metallicRoughness", mr_arr, True, tex_dir / f"{prefix}_metallicRoughness.png")
    n_arr = baked["normal"].copy()
    n_arr[..., 3] = 1.0
    normal_img = np_to_image(f"{prefix}_normal", n_arr, True, tex_dir / f"{prefix}_normal.png")
    results["maps"] = {"base_color": "baked", "metallic": "baked" if "metallic" in baked else "absent",
                       "roughness": "baked" if "roughness" in baked else "absent",
                       "alpha": "baked" if "alpha" in baked else "opaque", "normal": "baked_from_master"}
    return {"base": base_img, "mr": mr_img, "normal": normal_img, "info": results}


# --------------------------------------------------------------------------------------------- decimation

def weld(obj, diag: float):
    """Merge coincident vertices (UV-seam splits from the master export) so edge collapse is not blocked at seams.
    UVs are stored per loop, so seams survive; this only changes connectivity."""
    select_only([obj])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=diag * 1e-6, use_sharp_edge_from_normals=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def _welded_triangles(obj):
    """Triangles + welded positions (UV-seam splits merged) + per-vertex average normals, as numpy arrays."""
    me = obj.data
    me.calc_loop_triangles()
    n_tri = len(me.loop_triangles)
    tri = np.empty(n_tri * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    span = float(np.ptp(co, axis=0).max()) or 1.0
    q = np.round(co / span * 1e5).astype(np.int64)
    _, first, pos_id = np.unique(q, axis=0, return_index=True, return_inverse=True)
    v = np.ascontiguousarray(co[first], dtype=np.float32)
    f = pos_id.reshape(-1)[tri]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    return v, np.ascontiguousarray(f, dtype=np.uint32), np.ascontiguousarray(vn, dtype=np.float32)


def _replace_mesh(obj, v: np.ndarray, f: np.ndarray):
    me2 = bpy.data.meshes.new(obj.name + "_dec")
    me2.from_pydata(v.tolist(), [], f.tolist())
    me2.validate(verbose=False)
    me2.update()
    old = obj.data
    obj.data = me2
    if old.users == 0:
        bpy.data.meshes.remove(old)
    select_only([obj])
    bpy.ops.object.shade_smooth()
    if hasattr(bpy.ops.object, "shade_smooth_by_angle"):
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(45))


def _meshopt_simplify_with_attributes():
    """Bind meshopt_simplifyWithAttributes with its exact C signature (the PyPI wrapper leaves argtypes unset for it,
    which passes size_t as 32-bit ints and crashes). Signature per https://meshoptimizer.org/ :
    size_t meshopt_simplifyWithAttributes(unsigned int* destination, const unsigned int* indices, size_t index_count,
        const float* vertex_positions, size_t vertex_count, size_t vertex_positions_stride,
        const float* vertex_attributes, size_t vertex_attributes_stride, const float* attribute_weights, size_t attribute_count,
        const unsigned char* vertex_lock, size_t target_index_count, float target_error, unsigned int options, float* result_error)"""
    import ctypes

    from meshoptimizer._loader import lib

    fn = lib.meshopt_simplifyWithAttributes
    C = ctypes
    fn.argtypes = [C.POINTER(C.c_uint), C.POINTER(C.c_uint), C.c_size_t, C.POINTER(C.c_float), C.c_size_t, C.c_size_t,
                   C.POINTER(C.c_float), C.c_size_t, C.POINTER(C.c_float), C.c_size_t, C.POINTER(C.c_ubyte), C.c_size_t,
                   C.c_float, C.c_uint, C.POINTER(C.c_float)]
    fn.restype = C.c_size_t
    return fn, lib


def mo_decimate(obj, target: int, error_ladder=(0.01, 0.03, 0.05)) -> dict:
    """meshoptimizer (the engine-standard quadric simplifier), used as its docs recommend for whole-mesh LODs:
    vertices welded first (equivalent of meshopt_generateVertexRemap), attribute-aware with vertex normals
    (weight 0.7, docs: 0.5-1.0), meshopt_SimplifyPrune to drop isolated shells, and a relative target_error
    (fraction of the mesh extent) that is relaxed step by step only if the budget cannot be reached.
    Returns {"triangles", "result_error", "target_error", "reached"}; raises ImportError when the module is missing."""
    import ctypes

    import meshoptimizer as mo

    fn, _lib = _meshopt_simplify_with_attributes()
    v, f, vn = _welded_triangles(obj)
    idx = np.ascontiguousarray(f.reshape(-1), dtype=np.uint32)
    target_idx = int(target) * 3
    weights = np.array([0.7, 0.7, 0.7], dtype=np.float32)
    C = ctypes
    best = None
    for err in error_ladder:
        dest = np.empty(len(idx), dtype=np.uint32)
        res_err = C.c_float(0.0)
        n = fn(dest.ctypes.data_as(C.POINTER(C.c_uint)), idx.ctypes.data_as(C.POINTER(C.c_uint)), len(idx),
               v.ctypes.data_as(C.POINTER(C.c_float)), len(v), 12,
               vn.ctypes.data_as(C.POINTER(C.c_float)), 12, weights.ctypes.data_as(C.POINTER(C.c_float)), 3,
               None, target_idx, float(err), int(mo.SIMPLIFY_PRUNE), C.byref(res_err))
        best = (dest[:n].copy(), float(res_err.value), err)
        if n <= target_idx * 1.05:
            break
    new_idx, result_error, used_err = best
    used = np.unique(new_idx)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    _replace_mesh(obj, v[used], remap[new_idx].reshape(-1, 3))
    got = tri_count(obj)
    return {"triangles": got, "result_error_relative": result_error, "target_error_relative": used_err, "reached": got <= target * 1.05,
            "simplifier": "meshoptimizer.simplifyWithAttributes(normals w=0.7, PRUNE)"}


def lod_from_asset(asset, target: int, error_ladder=(0.01, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)) -> dict:
    """Simplify the optimized asset while preserving its UV layout (meshoptimizer with UV+normal attributes on the
    UV-split vertex set), so LODs reuse LOD0's textures and material: no re-unwrap, no re-bake, one texture set."""
    import ctypes

    import meshoptimizer as mo

    me = asset.data
    me.calc_loop_triangles()
    n_tri = len(me.loop_triangles)
    tri_loops = np.empty(n_tri * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("loops", tri_loops)
    loop_v = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loop_v)
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    # split vertices: unique (vertex, uv) pairs
    key = np.concatenate([loop_v[:, None].astype(np.float64), np.round(uv, 6).astype(np.float64)], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    v = np.ascontiguousarray(co[loop_v[first]], dtype=np.float32)
    vuv = np.ascontiguousarray(uv[first], dtype=np.float32)
    idx = np.ascontiguousarray(inv[tri_loops], dtype=np.uint32)
    # per split-vertex normals (area weighted) as extra attributes
    f = idx.reshape(-1, 3).astype(np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    attrs = np.ascontiguousarray(np.concatenate([vuv, vn], axis=1), dtype=np.float32)
    weights = np.array([10.0, 10.0, 0.5, 0.5, 0.5], dtype=np.float32)  # docs: texcoords 10-100, normals 0.5-1
    fn_, _ = _meshopt_simplify_with_attributes()
    C = ctypes
    target_idx = int(target) * 3
    best = None  # closest non-degenerate result; a too-loose error bound can collapse a small mesh completely
    for err in error_ladder:
        dest = np.empty(len(idx), dtype=np.uint32)
        res_err = C.c_float(0.0)
        n = fn_(dest.ctypes.data_as(C.POINTER(C.c_uint)), idx.ctypes.data_as(C.POINTER(C.c_uint)), len(idx),
                v.ctypes.data_as(C.POINTER(C.c_float)), len(v), 12,
                attrs.ctypes.data_as(C.POINTER(C.c_float)), 20, weights.ctypes.data_as(C.POINTER(C.c_float)), 5,
                None, target_idx, float(err), int(mo.SIMPLIFY_PRUNE), C.byref(res_err))
        # Loose error bounds on a small mesh can collapse it far below the target (or to nothing): only accept a
        # result that stays within reach of the target; otherwise keep the previous rung.
        if n < 36 or (err > 0.35 and n < target_idx * 0.5):
            break
        best = (dest[:n].copy(), float(res_err.value), err)
        if n <= target_idx * 1.05:
            break
    if best is None:
        best = (idx.copy(), 0.0, 0.0)
    new_idx, result_error, used_err = best
    used = np.unique(new_idx)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    faces = remap[new_idx].reshape(-1, 3)
    log(f"[lod] target={target} split_verts={len(v)} in_idx={len(idx)} out_idx={len(new_idx)} used={len(used)} err={result_error:.4f}@{used_err}")
    me2 = bpy.data.meshes.new(asset.name + "_lod")
    me2.from_pydata(v[used].tolist(), [], faces.tolist())
    me2.validate(verbose=False)
    uvl = me2.uv_layers.new(name="UVMap")
    lv = np.empty(len(me2.loops), dtype=np.int64)
    me2.loops.foreach_get("vertex_index", lv)
    uvl.data.foreach_set("uv", vuv[used][lv].reshape(-1))
    me2.update()
    lod = bpy.data.objects.new(asset.name + "_lod", me2)
    bpy.context.scene.collection.objects.link(lod)
    for m in asset.data.materials:
        lod.data.materials.append(m)
    select_only([lod])
    bpy.ops.object.shade_smooth()
    if hasattr(bpy.ops.object, "shade_smooth_by_angle"):
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(45))
    return {"object": lod, "triangles": tri_count(lod), "result_error_relative": result_error, "target_error_relative": used_err,
            "reached": tri_count(lod) <= target * 1.05, "simplifier": "meshoptimizer.simplifyWithAttributes(uv w=20, normals w=0.5, PRUNE) on LOD0"}


def reduce_to(obj, target: int, error_ladder=(0.01, 0.03, 0.05)) -> dict:
    """Reduction dispatcher: meshoptimizer within the error ladder; falls back to fast-simplification if the module is missing."""
    try:
        return mo_decimate(obj, target, error_ladder)
    except ImportError:
        n = fs_decimate(obj, target)
        return {"triangles": n, "reached": n <= target * 1.05, "simplifier": "fast-simplification (fallback)"}


def fs_decimate(obj, target: int) -> int:
    """Quadric edge-collapse decimation (fast-simplification) to an exact triangle count on the welded triangle soup.
    UVs are discarded (the caller re-unwraps and rebakes); this is only used on the rebake path."""
    import fast_simplification as fs

    me = obj.data
    me.calc_loop_triangles()
    n_tri = len(me.loop_triangles)
    tri = np.empty(n_tri * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    span = float(np.ptp(co, axis=0).max()) or 1.0
    q = np.round(co / span * 1e5).astype(np.int64)
    _, first, pos_id = np.unique(q, axis=0, return_index=True, return_inverse=True)
    v = co[first]
    f = pos_id.reshape(-1)[tri]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    v2, f2 = v.astype(np.float32), f.astype(np.int32)
    for agg in (7, 8, 9, 10):  # a welded soup can stall above the target; re-run with increasing aggressiveness
        v2, f2 = fs.simplify(v2, f2, target_count=int(target), agg=agg, verbose=False)
        v2, f2 = np.asarray(v2, dtype=np.float32), np.asarray(f2, dtype=np.int32)
        if len(f2) <= target * 1.02:
            break
    me2 = bpy.data.meshes.new(obj.name + "_dec")
    me2.from_pydata(v2.tolist(), [], f2.tolist())
    me2.validate(verbose=False)
    me2.update()
    old = obj.data
    obj.data = me2
    if old.users == 0:
        bpy.data.meshes.remove(old)
    select_only([obj])
    bpy.ops.object.shade_smooth()
    if hasattr(bpy.ops.object, "shade_smooth_by_angle"):
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(45))
    return tri_count(obj)


def voxel_proxy(obj, diag: float, voxel_frac: float = 0.0045) -> dict:
    """Single-shell proxy for heavy reductions: OpenVDB voxel remesh of the *copy* (never the master). Openings
    smaller than the voxel size are closed; larger openings (handles, windows, pipe gaps) survive."""
    voxel = max(diag * voxel_frac, 1e-4)
    select_only([obj])
    me = obj.data
    me.remesh_voxel_size = voxel
    me.remesh_voxel_adaptivity = 0.0
    me.use_remesh_fix_poles = False
    me.use_remesh_preserve_volume = True
    bpy.ops.object.voxel_remesh()
    return {"voxel_size_m": float(voxel), "proxy_triangles": tri_count(obj)}


def decimate(obj, ratio: float, delimit_uv: bool):
    if ratio >= 1.0:
        return
    mod = obj.modifiers.new("decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = float(ratio)
    mod.use_collapse_triangulate = True
    if delimit_uv:
        mod.delimit = {"UV"}
    tri = obj.modifiers.new("triangulate", "TRIANGULATE")
    tri.quad_method = "SHORTEST_DIAGONAL"
    apply_modifiers(obj)


def smart_uv(obj):
    select_only([obj])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.002, correct_aspect=True, scale_to_bounds=False)
    try:  # tighter packing = more texels on the surface (smart_project alone wastes most of the atlas on large meshes)
        bpy.ops.uv.pack_islands(rotate=False, margin=0.003, margin_method="FRACTION")
    except Exception:  # noqa: BLE001
        pass
    bpy.ops.object.mode_set(mode="OBJECT")


# --------------------------------------------------------------------------------------------- export / render

def export_glb(objs, path: Path, materials=True):
    select_only(objs)
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True, export_apply=True, export_yup=True,
        export_texcoords=True, export_normals=True, export_tangents=False,  # engines generate MikkTSpace tangents at import; exported ones can be degenerate
        export_materials="EXPORT" if materials else "NONE", export_image_format="AUTO",
        export_animations=False, export_skins=False, export_morph=False, export_lights=False, export_cameras=False,
        export_extras=False,
    )
    return {"file": path.name, "bytes": path.stat().st_size}


def setup_render(size: int):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 64
    sc.cycles.use_denoising = True
    sc.render.resolution_x = size
    sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.62, 0.63, 0.65, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    for name, rot, strength in (("key", (math.radians(50), 0, math.radians(35)), 3.0),
                                ("fill", (math.radians(60), 0, math.radians(-120)), 1.2),
                                ("rim", (math.radians(35), 0, math.radians(160)), 1.5)):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = strength
        ld.angle = math.radians(12)
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = rot
        sc.collection.objects.link(lo)
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = 50
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def render_views(obj, cam, out_dir: Path, prefix: str, views) -> list[str]:
    mn, mx = bounds(obj)
    center = Vector(((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2))
    r = float(np.linalg.norm(mx - mn)) / 2
    fov = cam.data.angle
    d = r / math.sin(fov / 2) * 1.15
    files = []
    for name, az, el in views:
        az_r, el_r = math.radians(az), math.radians(el)
        pos = center + Vector((d * math.cos(el_r) * math.sin(az_r), -d * math.cos(el_r) * math.cos(az_r), d * math.sin(el_r)))
        cam.location = pos
        direction = center - pos
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        f = out_dir / f"{prefix}_{name}.png"
        bpy.context.scene.render.filepath = str(f)
        bpy.ops.render.render(write_still=True)
        files.append(f.name)
    return files


def contact_sheet(out_dir: Path, files: list[str], reference: str | None, size: int):
    try:
        from PIL import Image
    except ImportError:
        return None
    ims = [Image.open(out_dir / f).convert("RGB").resize((size, size)) for f in files]
    if reference and Path(reference).exists():
        ims.insert(0, Image.open(reference).convert("RGB").resize((size, size)))
    n = len(ims)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * size, rows * size), (40, 40, 40))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * size, (i // cols) * size))
    p = out_dir / "contact_sheet.png"
    sheet.save(p)
    return p.name


def mesh_report(obj, note=None) -> dict:
    mn, mx = bounds(obj)
    rep = {"vertices": len(obj.data.vertices), "triangles": tri_count(obj), "has_uv": bool(obj.data.uv_layers),
           "bounds_min": [float(x) for x in mn], "bounds_max": [float(x) for x in mx],
           "dimensions_m": [float(x) for x in (mx - mn)]}
    if note:
        rep["note"] = note
    return rep


# --------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    a = ap.parse_args()
    req = json.loads(Path(a.request).read_text(encoding="utf-8"))
    opt = req["optimize"]
    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = req.get("name", "asset")
    t_all = time.time()
    files: list[str] = []
    outputs: dict = {}

    reset_scene()
    log(f"bpy {bpy.app.version_string}; importing {req['master_glb']}")
    t0 = time.time()
    master = import_master(req["master_glb"])
    tick("import_s", t0)
    STATS["master_imported"] = mesh_report(master)

    t0 = time.time()
    STATS["normalise"] = normalise(master, opt)
    STATS["cleanup"] = cleanup(master)
    tick("normalise_cleanup_s", t0)
    mn, mx = bounds(master)
    diag = float(np.linalg.norm(mx - mn)) or 1.0
    maps = master_maps(master)
    ginfo = glb_material_info(req["master_glb"])
    maps["alpha_mode"] = ginfo.get("alpha_mode", "OPAQUE")
    STATS["master_material"] = ginfo
    STATS["master_maps"] = {k: (f"{v.size[0]}x{v.size[1]}" if v is not None else None) for k, v in maps["images"].items()}

    # ---- optimized asset ------------------------------------------------------------------------
    t0 = time.time()
    target = int(opt["target_triangles"])
    tex_size = int(opt["texture_size"])
    cur = tri_count(master)
    ratio = target / max(cur, 1)
    asset = duplicate(master, "asset")
    weld(asset, diag)
    tex_dir = out_dir / "textures"
    tex_dir.mkdir(exist_ok=True)
    method = None
    proxy_info = None
    reduce_info = None
    if ratio >= 1.0:
        method = "no_decimation"
        tri = asset.modifiers.new("triangulate", "TRIANGULATE")
        apply_modifiers(asset)
    elif ratio >= 0.35:
        decimate(asset, ratio, True)
        got = tri_count(asset)
        if got > target * 1.05:
            log(f"UV-delimited decimation reached {got} > {target}; falling back to rebake path")
            bpy.data.objects.remove(asset, do_unlink=True)
            asset = duplicate(master, "asset")
            reduce_info = reduce_to(asset, target)
            method = "decimate_rebake"
        else:
            method = "decimate_keep_uv"
    else:
        reduce_info = reduce_to(asset, target)
        method = "decimate_rebake"
        if not reduce_info["reached"]:
            # topology-limited (many thin shells): rebuild a single-shell proxy of the copy and reduce that instead
            log(f"quadric simplification could not reach {target} within 5% error (got {reduce_info['triangles']}); using a voxel proxy")
            bpy.data.objects.remove(asset, do_unlink=True)
            asset = duplicate(master, "asset")
            weld(asset, diag)
            proxy_info = voxel_proxy(asset, diag)
            WARN.append(f"reduction to {ratio:.2%} of the master exceeded a 5% error bound (thin/many shells): optimized mesh built "
                        f"from a single-shell voxel proxy (voxel {proxy_info['voxel_size_m']*1000:.1f} mm); openings smaller than that "
                        f"are closed in the low-poly, detail comes from the baked normal/colour maps")
            reduce_info = reduce_to(asset, target, error_ladder=(0.03, 0.05, 0.2, 1.0))
            method = "voxel_proxy_decimate_rebake"
    got = tri_count(asset)
    renormalise(asset, opt)
    lod_base = duplicate(asset, "lod_base")  # reduced geometry (proxy/decimated) before UV/material work; LODs derive from it
    lod_base.hide_render = True
    asset_info = {"method": method, "master_triangles": cur, "target_triangles": target, "triangles": got, "proxy": proxy_info,
                  "reduction": reduce_info}
    if method in ("decimate_rebake", "voxel_proxy_decimate_rebake"):
        smart_uv(asset)
        cage = max(diag * 0.01, 2.5 * proxy_info["voxel_size_m"]) if proxy_info else None
        baked = bake_from_master(master, asset, tex_size, maps, out_dir, diag, cage=cage)
        mat = build_pbr_material("asset_material", baked["base"], baked["mr"], baked["normal"], maps["alpha_mode"])
        asset_info.update(baked["info"])
        asset_info["uv"] = "smart_uv_project (new)"
    else:
        base = maps["images"]["base_color"]
        mr = maps["images"]["metallic"] or maps["images"]["roughness"]
        if base is None:
            raise RuntimeError("master has no base colour texture")
        base_img = resized_copy(base, tex_size, "asset_baseColor", tex_dir / "asset_baseColor.png", False)
        mr_img = resized_copy(mr, tex_size, "asset_metallicRoughness", tex_dir / "asset_metallicRoughness.png", True) if mr is not None else None
        if mr_img is None:
            arr = np.zeros((tex_size, tex_size, 4), dtype=np.float32)
            arr[..., 1] = 0.5
            arr[..., 3] = 1.0
            mr_img = np_to_image("asset_metallicRoughness", arr, True, tex_dir / "asset_metallicRoughness.png")
        mat = build_pbr_material("asset_material", base_img, mr_img, None, maps["alpha_mode"])
        asset_info["maps"] = {"base_color": "reused_from_master" + ("_resized" if base.size[0] != tex_size else ""),
                              "metallic": "reused_from_master" if mr is not None else "absent",
                              "roughness": "reused_from_master" if mr is not None else "absent",
                              "alpha": "reused_from_master" if maps["alpha_mode"] != "OPAQUE" else "opaque",
                              "normal": "none (geometry retained; no map baked)"}
        asset_info["uv"] = "master UVs preserved"
    asset.data.materials.clear()
    asset.data.materials.append(mat)
    asset.name = name
    p = out_dir / f"{name}.glb"
    export_glb([asset], p)
    files.append(p.name)
    outputs["asset_glb"] = p.name
    asset_info.update(mesh_report(asset))
    STATS["asset"] = asset_info
    tick("optimize_s", t0)
    log(f"optimized asset: {asset_info['triangles']} tris via {method}")

    # ---- LODs ------------------------------------------------------------------------------------
    outputs["lods"] = []
    if opt.get("generate_lods"):
        t0 = time.time()
        prev = asset  # LODs form a chain (docs: simplifying the previous LOD is cheaper and gives smoother transitions)
        for i, frac in enumerate(opt.get("lod_fractions", [0.5, 0.25]), start=1):
            lod_target = max(12, int(got * frac))
            info = lod_from_asset(prev, lod_target)
            lod = info.pop("object")
            lod.name = f"{name}_LOD{i}"
            renormalise(lod, opt)
            p = out_dir / f"{name}_LOD{i}.glb"
            export_glb([lod], p)
            files.append(p.name)
            outputs["lods"].append({"file": p.name, "fraction": frac, "target_triangles": lod_target,
                                    "maps": "shared_with_asset (UVs preserved)", "reduction": info, **mesh_report(lod)})
            if info["triangles"] > lod_target * 1.1:
                WARN.append(f"LOD{i} stopped at {info['triangles']} triangles (budget {lod_target}): further UV-preserving "
                            f"collapse would collapse the mesh (small budgets: LODs are best-effort)")
            if prev is not asset:
                bpy.data.objects.remove(prev, do_unlink=True)
            prev = lod
        if prev is not asset:
            bpy.data.objects.remove(prev, do_unlink=True)
        tick("lods_s", t0)

    bpy.data.objects.remove(lod_base, do_unlink=True)

    # ---- collision -------------------------------------------------------------------------------
    outputs["collision_glb"] = None
    if opt.get("generate_collision"):
        t0 = time.time()
        col = duplicate(asset, f"{name}_collision")
        col.data.materials.clear()
        bm = bmesh.new()
        bm.from_mesh(col.data)
        bmesh.ops.delete(bm, geom=bm.faces[:], context="FACES_ONLY")
        bmesh.ops.delete(bm, geom=bm.edges[:], context="EDGES_FACES")
        hull = bmesh.ops.convex_hull(bm, input=bm.verts[:])
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
        bm.to_mesh(col.data)
        bm.free()
        col.data.update()
        ct = int(opt.get("collision_triangles", 200))
        decimate(col, ct / max(tri_count(col), 1), False)
        p = out_dir / f"{name}_collision.glb"
        export_glb([col], p, materials=False)
        files.append(p.name)
        outputs["collision_glb"] = p.name
        STATS["collision"] = {"type": "convex_hull_decimated", "target_triangles": ct, **mesh_report(col)}
        bpy.data.objects.remove(col, do_unlink=True)
        tick("collision_s", t0)

    # ---- previews --------------------------------------------------------------------------------
    outputs["previews"] = []
    if opt.get("render_previews", True):
        t0 = time.time()
        prev = out_dir / "previews"
        prev.mkdir(exist_ok=True)
        cam = setup_render(int(opt.get("preview_size", 512)))
        master.hide_render = True
        asset.hide_render = False
        views = [("front_left", 35, 22), ("front_right", -35, 22), ("back", 200, 18), ("top", 35, 65)]
        fs = render_views(asset, cam, prev, "asset", views)
        asset.hide_render = True
        master.hide_render = False
        fs += render_views(master, cam, prev, "master", [("front_left", 35, 22)])
        sheet = contact_sheet(prev, fs, req.get("reference_png"), int(opt.get("preview_size", 512)))
        if sheet:
            fs.append(sheet)
        outputs["previews"] = [f"previews/{f}" for f in fs]
        files.extend(outputs["previews"])
        tick("previews_s", t0)

    # loose texture maps (also embedded in the GLBs) for engines/tools that want separate files
    for tp in sorted((out_dir / "textures").glob("*.png")):
        files.append(f"textures/{tp.name}")
    outputs["textures"] = [f for f in files if f.startswith("textures/")]
    T["total_s"] = round(time.time() - t_all, 2)
    STATS["timings_s"] = T
    STATS["asset_kind"] = "optimized static asset (automatic decimation); not animation-ready topology, no rig"
    result = {"status": "ok", "files": files, "outputs": outputs, "stats": STATS, "warnings": WARN}
    tmp = out_dir / "output.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, out_dir / "output.json")
    log(f"done in {T['total_s']}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
