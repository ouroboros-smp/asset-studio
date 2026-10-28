"""Structural validation of GLB files: meshes, triangle counts, UVs, normals, materials, textures, extensions,
bounds, origin and scale. Pure-python (pygltflib + struct); no engine needed. Findings are facts, not claims:
a map is reported present only if the material actually references an image.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pygltflib

_COMPONENT = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _read_accessor(g: pygltflib.GLTF2, blob: bytes, idx: int) -> np.ndarray:
    acc = g.accessors[idx]
    bv = g.bufferViews[acc.bufferView]
    fmt, size = _COMPONENT[acc.componentType]
    n = _NCOMP[acc.type]
    stride = bv.byteStride or size * n
    base = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    out = np.empty((acc.count, n), dtype=np.dtype(fmt))
    if stride == size * n:
        out = np.frombuffer(blob, dtype=np.dtype(fmt), count=acc.count * n, offset=base).reshape(acc.count, n)
    else:
        for i in range(acc.count):
            out[i] = struct.unpack_from("<" + fmt * n, blob, base + i * stride)
    return out


def _png_size(data: bytes):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return w, h
    if data[:3] == b"\xff\xd8\xff":  # JPEG: scan for SOF
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            seg = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + seg
    return None


def validate_glb(path: str | Path, expect: dict | None = None) -> dict:
    """Return a report dict with 'ok', 'errors', 'warnings', and measured statistics."""
    expect = expect or {}
    path = Path(path)
    rep = {"file": path.name, "bytes": path.stat().st_size, "errors": [], "warnings": [], "ok": True}
    try:
        g = pygltflib.GLTF2().load(str(path))
    except Exception as e:  # noqa: BLE001
        rep["errors"].append(f"not a loadable glTF: {e}")
        rep["ok"] = False
        return rep
    blob = g.binary_blob() or b""
    rep["extensions_required"] = list(g.extensionsRequired or [])
    rep["extensions_used"] = list(g.extensionsUsed or [])
    if "EXT_texture_webp" in rep["extensions_required"]:
        rep["errors"].append("EXT_texture_webp is required: not portable to engines without WebP support")
    # images
    images = []
    for i, im in enumerate(g.images or []):
        entry = {"index": i, "mimeType": im.mimeType, "name": im.name}
        if im.bufferView is not None:
            bv = g.bufferViews[im.bufferView]
            data = blob[bv.byteOffset or 0:(bv.byteOffset or 0) + bv.byteLength]
            entry["bytes"] = len(data)
            entry["size"] = _png_size(data)
            if im.mimeType not in ("image/png", "image/jpeg"):
                rep["errors"].append(f"image {i} has non-standard mimeType {im.mimeType}")
        else:
            rep["errors"].append(f"image {i} is an external URI, expected embedded")
        images.append(entry)
    rep["images"] = images
    # materials
    mats = []
    for i, m in enumerate(g.materials or []):
        pbr = m.pbrMetallicRoughness
        entry = {"index": i, "name": m.name, "alphaMode": m.alphaMode, "doubleSided": m.doubleSided,
                 "baseColorTexture": None, "metallicRoughnessTexture": None, "normalTexture": None,
                 "metallicFactor": pbr.metallicFactor if pbr else None, "roughnessFactor": pbr.roughnessFactor if pbr else None}
        if pbr and pbr.baseColorTexture is not None:
            entry["baseColorTexture"] = g.textures[pbr.baseColorTexture.index].source
        if pbr and pbr.metallicRoughnessTexture is not None:
            entry["metallicRoughnessTexture"] = g.textures[pbr.metallicRoughnessTexture.index].source
        if m.normalTexture is not None:
            entry["normalTexture"] = g.textures[m.normalTexture.index].source
        mats.append(entry)
    rep["materials"] = mats
    # meshes
    meshes = []
    tri_total = 0
    vert_total = 0
    mins, maxs = [], []
    for mi, mesh in enumerate(g.meshes or []):
        for pi, prim in enumerate(mesh.primitives):
            attrs = prim.attributes
            pos = _read_accessor(g, blob, attrs.POSITION).astype(np.float32)
            nverts = len(pos)
            if prim.indices is not None:
                nidx = g.accessors[prim.indices].count
            else:
                nidx = nverts
            mode = prim.mode if prim.mode is not None else 4
            tris = nidx // 3 if mode == 4 else 0
            has_uv = attrs.TEXCOORD_0 is not None
            has_n = attrs.NORMAL is not None
            has_t = attrs.TANGENT is not None
            uv_ok = None
            if has_uv:
                uv = _read_accessor(g, blob, attrs.TEXCOORD_0)
                uv_ok = bool(np.isfinite(uv).all())
                if not uv_ok:
                    rep["errors"].append(f"mesh {mi} prim {pi}: non-finite UVs")
            if not np.isfinite(pos).all():
                rep["errors"].append(f"mesh {mi} prim {pi}: non-finite positions")
            mins.append(pos.min(0))
            maxs.append(pos.max(0))
            tri_total += tris
            vert_total += nverts
            meshes.append({"mesh": mi, "name": mesh.name, "primitive": pi, "vertices": nverts, "triangles": tris,
                           "mode": mode, "has_uv": has_uv, "has_normals": has_n, "has_tangents": has_t,
                           "material": prim.material})
    rep["meshes"] = meshes
    rep["triangles"] = tri_total
    rep["vertices"] = vert_total
    rep["mesh_count"] = len(g.meshes or [])
    rep["node_count"] = len(g.nodes or [])
    if not meshes:
        rep["errors"].append("no mesh primitives")
    if mins:
        # apply node transforms only if trivial; report a warning otherwise
        mn = np.min(np.stack(mins), 0)
        mx = np.max(np.stack(maxs), 0)
        nontrivial = False
        for n in g.nodes or []:
            if (n.matrix and n.matrix != [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) or n.scale not in (None, [1, 1, 1]) \
                    or n.translation not in (None, [0, 0, 0]) or n.rotation not in (None, [0, 0, 0, 1]):
                nontrivial = True
        if nontrivial:
            rep["warnings"].append("node transforms are not identity; bounds below are mesh-local")
        rep["bounds_min"] = [float(x) for x in mn]
        rep["bounds_max"] = [float(x) for x in mx]
        dims = (mx - mn)
        rep["dimensions_xyz"] = [float(x) for x in dims]
        rep["height_y"] = float(dims[1])
        rep["origin"] = {"bottom_center": bool(abs(mn[1]) < 1e-3 * max(1.0, dims[1]) and
                                               abs((mn[0] + mx[0]) / 2) < 0.02 * max(dims[0], 1e-6) + 1e-4 and
                                               abs((mn[2] + mx[2]) / 2) < 0.02 * max(dims[2], 1e-6) + 1e-4)}
    # expectations
    if "max_triangles" in expect and tri_total > expect["max_triangles"]:
        rep["errors"].append(f"triangles {tri_total} exceed budget {expect['max_triangles']}")
    if "soft_max_triangles" in expect and tri_total > expect["soft_max_triangles"]:
        # LODs: the reducer stops before the target when further UV-preserving collapse would distort the mesh;
        # that is reported, not fatal (the LOD0 asset itself is held to its budget strictly)
        rep["warnings"].append(f"triangles {tri_total} exceed budget {expect['soft_max_triangles']} (reducer stopped early)")
    if expect.get("require_uv") and any(not m["has_uv"] for m in meshes):
        rep["errors"].append("a primitive has no TEXCOORD_0")
    if expect.get("require_material") and (not mats or any(m["material"] is None for m in meshes)):
        rep["errors"].append("a primitive has no material")
    if expect.get("require_basecolor") and not any(m["baseColorTexture"] is not None for m in mats):
        rep["errors"].append("no material has a baseColorTexture")
    if "height_m" in expect and expect["height_m"] and mins:
        h = rep["height_y"]
        if abs(h - expect["height_m"]) > 0.02 * expect["height_m"]:
            rep["errors"].append(f"height {h:.4f} m differs from requested {expect['height_m']} m")
    if "max_texture" in expect:
        for im in images:
            if im.get("size") and max(im["size"]) > expect["max_texture"]:
                rep["errors"].append(f"image {im['index']} is {im['size']} > {expect['max_texture']}")
    if expect.get("require_bottom_origin") and mins and not rep["origin"]["bottom_center"]:
        rep["errors"].append("origin is not bottom-centre")
    rep["ok"] = not rep["errors"]
    return rep


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(validate_glb(sys.argv[1]), indent=2))
