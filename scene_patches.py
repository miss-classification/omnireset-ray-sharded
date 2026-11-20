"""Worker-side env_cfg patches for experiments. Each takes env_cfg (after hydra compose, before gym.make), mutates
it, and returns a small dict describing what it did. Runs inside the actor (Kit already booted, pxr available).

shared_object_materials: the 64K PhysX material ceiling comes from the PhysicsMaterial prim that peg.usd and
peg_hole.usd each carry; Isaac Lab clones it into every env (2 per env). This writes a flattened local copy of each
object USD without its physics-material prims / bindings, so the objects use the scene's default material (as the
robot and table already do), and points the scene cfg at the copies. Friction is still randomized per shape at
startup by the task's bucketed randomize_rigid_body_material terms (PhysX tensor API), which is what OmniReset
trains with; only the pre-randomization default values change. Validated by expert success, not assumed.
"""
import hashlib
import os


def _strip_physics_materials(src_url, out_dir):
    from pxr import Usd, UsdPhysics, UsdShade
    stage = Usd.Stage.Open(src_url)
    if stage is None:
        raise RuntimeError(f"cannot open {src_url}")
    flat = Usd.Stage.Open(stage.Flatten())
    removed, unbound, props = [], 0, []
    for prim in list(flat.Traverse()):
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            m = UsdPhysics.MaterialAPI(prim)
            props.append(dict(path=str(prim.GetPath()),
                              static=m.GetStaticFrictionAttr().Get(), dynamic=m.GetDynamicFrictionAttr().Get(),
                              restitution=m.GetRestitutionAttr().Get()))
            removed.append(str(prim.GetPath()))
    for prim in flat.Traverse():
        for rel in prim.GetRelationships():
            if rel.GetName().startswith("material:binding") and any(
                    str(t) in removed or any(str(t).startswith(r + "/") for r in removed) for t in rel.GetTargets()):
                rel.ClearTargets(True)
                prim.RemoveProperty(rel.GetName())
                unbound += 1
    for path in removed:
        # if the material prim is also a render material holder, only drop the physics API, else remove the prim
        prim = flat.GetPrimAtPath(path)
        if prim and prim.IsValid():
            if prim.IsA(UsdShade.Material) and prim.GetChildren():
                prim.RemoveAPI(UsdPhysics.MaterialAPI)
            else:
                flat.RemovePrim(path)
    # UWLab derives the reset-dataset pair name from the USD's parent directory name (Peg, PegHole) and reads
    # metadata.yaml next to the USD -> keep the directory name and copy the metadata alongside.
    parent = src_url.rstrip("/").split("/")[-2]
    d = os.path.join(out_dir, hashlib.md5(src_url.encode()).hexdigest()[:8], parent)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(src_url))
    flat.GetRootLayer().Export(dst)
    import urllib.request
    meta_src = src_url.rsplit("/", 1)[0] + "/metadata.yaml"
    try:
        urllib.request.urlretrieve(meta_src, os.path.join(d, "metadata.yaml"))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"metadata.yaml not retrievable from {meta_src}: {e}") from e
    return dict(src=src_url, dst=dst, removed_material_prims=removed, removed_bindings=unbound, original_props=props)


def shared_object_materials(env_cfg, out_dir="/tmp/shared_mat_usd"):
    info = {}
    for key in ("insertive_object", "receptive_object"):
        obj = getattr(env_cfg.scene, key)
        info[key] = _strip_physics_materials(obj.spawn.usd_path, out_dir)
        obj.spawn.usd_path = info[key]["dst"]
    return info


def count_physics_materials(num_envs):
    """USD prims with UsdPhysics.MaterialAPI in env_0 (x num_envs) + outside /World/envs."""
    import omni.usd
    from pxr import Usd, UsdPhysics
    stage = omni.usd.get_context().get_stage()

    def count(path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return []
        return [str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.MaterialAPI)]
    env0 = count("/World/envs/env_0")
    glob = []
    for c in stage.GetPseudoRoot().GetChildren():
        if c.GetName() != "World":
            glob += count(str(c.GetPath()))
    world = stage.GetPrimAtPath("/World")
    for c in (world.GetChildren() if world else []):
        if c.GetName() != "envs":
            glob += count(str(c.GetPath()))
    return dict(per_env=len(env0), env0_paths=env0, global_=len(glob), global_paths=glob,
                usd_total_est=len(env0) * num_envs + len(glob))


def vmhwm_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 2**20
    return None
