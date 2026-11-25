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


# ---------------- physics-start (PhysX parse) optimizations ----------------
# Measured at 32,000 envs / 1 process: force_load_physics_from_usd = 631 s of 1,121 s startup. In Isaac Lab
# 3.0.0-EA the PhysX replicator runs inside that call with useEnvIds=False ("TODO: envIds needs to support
# heterogeneous setup. for now, we rely on USD collision filtering") and the scene authors one USD
# PhysicsCollisionGroup per env (+ a global group listing all of them) for cross-env filtering.
# These patches put back PhysX's env-ID path (envs are isolated by env id on the GPU; no collision groups).

def _patch_physx_replicator(use_env_ids=True, use_fabric=False):
    from isaaclab_physx.cloner import replicate as R
    if getattr(R.PhysxReplicateContext.replicate, "_patched", None) == (use_env_ids, use_fabric):
        return
    from omni.physx import get_physx_replicator_interface

    def replicate(self):
        if not self._queue:
            return
        physx_queue = tuple(self._queue)
        self._queue.clear()
        current_worlds, cur = [], {"t": ""}

        def attach_fn(_sid):
            return ["/World/template", "/World/envs"]

        def rename_fn(_p, i):
            return cur["t"].format(current_worlds[i])

        def attach_end_fn(_sid):
            rep = get_physx_replicator_interface()
            for src, destination, target_envs in physx_queue:
                cur["t"] = destination
                current_worlds[:] = target_envs
                if not current_worlds:
                    continue
                rep.replicate(_sid, src, len(current_worlds), useEnvIds=use_env_ids,
                              useFabricForReplication=use_fabric)
            rep.unregister_replicator(_sid)

        get_physx_replicator_interface().register_replicator(self._stage_id, attach_fn, attach_end_fn, rename_fn)
    replicate._patched = (use_env_ids, use_fabric)
    R.PhysxReplicateContext.replicate = replicate


def envids_nofilter(env_cfg):
    """PhysX env-ID isolation (useEnvIds=True) instead of per-env USD collision groups."""
    env_cfg.scene.filter_collisions = False
    _patch_physx_replicator(use_env_ids=True)
    return dict(patch="envids_nofilter", filter_collisions=False, use_env_ids=True)


def nofilter(env_cfg):
    """No cross-env filtering at all (cost of the collision groups alone; envs are 1.5 m apart)."""
    env_cfg.scene.filter_collisions = False
    return dict(patch="nofilter", filter_collisions=False, use_env_ids=False)


def envids_fabric_nofilter(env_cfg):
    env_cfg.scene.filter_collisions = False
    _patch_physx_replicator(use_env_ids=True, use_fabric=True)
    return dict(patch="envids_fabric_nofilter", filter_collisions=False, use_env_ids=True, use_fabric=True)


def envids_nofilter_shared_mat(env_cfg, out_dir="/tmp/shared_mat_usd"):
    info = envids_nofilter(env_cfg)
    info["materials"] = shared_object_materials(env_cfg, out_dir)
    return info


# ---------------- asset format: flattened .usda instead of compressed .usd crate ----------------
# Native stacks of force_load_physics_from_usd at 8,192 envs: ~40% of samples read VtArray<int> (mesh topology) from
# the referenced asset crate files, i.e. decompressed again for every env. A flattened .usda layer holds values as
# in-memory VtValues (shared, no per-read decompression). Physics content is identical (same prims / attributes).

def _flatten_export(src_url, out_dir, ext=".usda", copy_metadata=False):
    from pxr import Usd
    stage = Usd.Stage.Open(src_url)
    if stage is None:
        raise RuntimeError(f"cannot open {src_url}")
    parent = src_url.rstrip("/").split("/")[-2]
    d = os.path.join(out_dir, hashlib.md5(src_url.encode()).hexdigest()[:8], parent)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.splitext(os.path.basename(src_url))[0] + ext)
    stage.Flatten().Export(dst)
    if copy_metadata:
        meta_src = src_url.rsplit("/", 1)[0] + "/metadata.yaml"
        if os.path.exists(meta_src):
            import shutil
            shutil.copy(meta_src, os.path.join(d, "metadata.yaml"))
        else:
            import urllib.request
            try:
                urllib.request.urlretrieve(meta_src, os.path.join(d, "metadata.yaml"))
            except Exception:  # noqa: BLE001  (not every asset ships one; UWLab reads it for robot + objects)
                pass
    return dst


def usda_assets(env_cfg, out_dir="/tmp/usda_assets"):
    """Every scene asset spawned from a USD file -> flattened local .usda copy."""
    import time as _t
    info = {}
    for name in list(vars(env_cfg.scene)):
        obj = getattr(env_cfg.scene, name, None)
        spawn = getattr(obj, "spawn", None)
        path = getattr(spawn, "usd_path", None)
        if not path or not str(path).lower().endswith((".usd", ".usdc", ".usdz")):
            continue
        t = _t.time()
        dst = _flatten_export(path, out_dir, ".usda", copy_metadata=True)
        spawn.usd_path = dst
        info[name] = dict(src=path, dst=dst, export_s=round(_t.time() - t, 2), bytes=os.path.getsize(dst))
    return info


def usdc_flat_assets(env_cfg, out_dir="/tmp/usdc_flat_assets"):
    """Control for usda_assets: same flattening, written back as a crate (.usd) -> isolates format from flattening."""
    info = {}
    for name in list(vars(env_cfg.scene)):
        obj = getattr(env_cfg.scene, name, None)
        spawn = getattr(obj, "spawn", None)
        path = getattr(spawn, "usd_path", None)
        if not path or not str(path).lower().endswith((".usd", ".usdc", ".usdz")):
            continue
        dst = _flatten_export(path, out_dir, ".usd", copy_metadata=True)
        spawn.usd_path = dst
        info[name] = dict(src=path, dst=dst)
    return info


def usda_assets_shared_mat(env_cfg):
    """usda + the 64K-material fix (objects without per-env physics materials)."""
    info = dict(materials=shared_object_materials(env_cfg))
    info["usda"] = usda_assets(env_cfg)
    return info


def newton_no_usd_clones(env_cfg):
    """Newton only: skip the per-env USD prim copies (UsdReplicateContext) that Isaac Lab adds whenever Kit is running.
    Newton builds the physics model from env_0 + ModelBuilder.replicate, so headless training does not need USD clones
    (Isaac Lab's kit-less Newton path never makes them). Measured: ~100 s of Newton's 123 s scene creation at 32K envs."""
    import isaaclab.cloner.replicate_session as rs
    rs.has_kit = lambda: False
    return dict(patch="newton_no_usd_clones")


def _patch_no_fabric_no_usd_sync():
    """Headless physics-only: no omni.physx.fabric (Fabric/USDRT population of every env) AND no write-back to USD.
    Isaac Lab reads all state from PhysX tensor views, so training does not need either. (Isaac Lab's own
    use_fabric=False instead turns ON USD write-back every step.)"""
    from isaaclab_physx.physics import physx_manager as pm
    M = pm.PhysxManager
    if getattr(M._load_fabric, "_patched", False):
        return

    @classmethod
    def _load_fabric(cls):
        import omni.kit.app
        sim = pm.PhysicsManager._sim
        if sim is None:
            return
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        if ext_mgr.is_extension_enabled("omni.physx.fabric"):
            ext_mgr.set_extension_enabled_immediate("omni.physx.fabric", False)
        cls._fabric = None
        cls._update_fabric = None
        for key in ["updateToUsd", "updateParticlesToUsd", "updateVelocitiesToUsd", "updateForceSensorsToUsd",
                    "updateResidualsToUsd"]:
            sim.set_setting(f"/physics/{key}", False)
        sim.set_setting("/isaaclab/fabric_enabled", False)
        sim.set_setting("/physics/visualizationDisplaySimulationOutput", False)
    _load_fabric.__func__._patched = True
    M._load_fabric = _load_fabric


def usda_shared_mat_nofabric(env_cfg):
    info = usda_assets_shared_mat(env_cfg)
    _patch_no_fabric_no_usd_sync()
    info["no_fabric_no_usd_sync"] = True
    return info


# ---------------- physics-only assets (headless training) ----------------
def _physics_only_flatten(src_url, out_dir):
    """Flatten, then drop render-only content: Gprims with no CollisionAPI on themselves or an ancestor, UsdShade
    Materials/Shaders without PhysicsMaterialAPI, and material bindings that do not target a physics material.
    Collision geometry, rigid bodies, joints, articulation roots, mass and physics materials stay untouched."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.Open(src_url)
    flat = Usd.Stage.Open(stage.Flatten())
    flat.SetEditTarget(flat.GetRootLayer())
    phys_mats, removed_gprims, removed_mats, removed_bind = set(), 0, 0, 0
    for prim in Usd.PrimRange.Stage(flat, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            phys_mats.add(str(prim.GetPath()))

    def has_collision(p):
        while p and p.IsValid() and not p.IsPseudoRoot():
            if p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.MeshCollisionAPI):
                return True
            p = p.GetParent()
        return False
    to_remove = []
    for prim in flat.Traverse():
        if prim.IsInstanceProxy():
            continue
        if prim.IsA(UsdGeom.Gprim) and not has_collision(prim) and not prim.GetChildren():
            to_remove.append(prim.GetPath())
            removed_gprims += 1
        elif (prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader)) and str(prim.GetPath()) not in phys_mats \
                and not any(str(prim.GetPath()).startswith(pm + "/") for pm in phys_mats):
            if prim.IsA(UsdShade.Material):
                to_remove.append(prim.GetPath())
                removed_mats += 1
        for rel in prim.GetRelationships():
            if rel.GetName().startswith("material:binding") and not rel.GetName().startswith("material:binding:physics"):
                tg = [str(t) for t in rel.GetTargets()]
                if not any(t in phys_mats for t in tg):
                    prim.RemoveProperty(rel.GetName())
                    removed_bind += 1
    # Flatten() writes instance prototypes as ordinary root prims (/Flattened_Prototype_*), so Traverse() covers them.
    for path in sorted(set(to_remove), key=lambda p: -len(str(p))):
        flat.RemovePrim(path)
    parent = src_url.rstrip("/").split("/")[-2]
    d = os.path.join(out_dir, hashlib.md5(src_url.encode()).hexdigest()[:8], parent)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.splitext(os.path.basename(src_url))[0] + ".usda")
    flat.GetRootLayer().Export(dst)
    meta_src = src_url.rsplit("/", 1)[0] + "/metadata.yaml"
    try:
        if os.path.exists(meta_src):
            import shutil
            shutil.copy(meta_src, os.path.join(d, "metadata.yaml"))
        else:
            import urllib.request
            urllib.request.urlretrieve(meta_src, os.path.join(d, "metadata.yaml"))
    except Exception:  # noqa: BLE001
        pass
    return dict(src=src_url, dst=dst, removed_gprims=removed_gprims, removed_materials=removed_mats,
                removed_bindings=removed_bind, kept_physics_materials=len(phys_mats), bytes=os.path.getsize(dst))


def physics_only_shared_mat(env_cfg, out_dir="/tmp/physics_only_assets"):
    """Shared physics material on the objects (64K fix), then every scene asset -> physics-only flattened .usda."""
    info = dict(materials=shared_object_materials(env_cfg))
    for name in list(vars(env_cfg.scene)):
        obj = getattr(env_cfg.scene, name, None)
        spawn = getattr(obj, "spawn", None)
        path = getattr(spawn, "usd_path", None)
        if not path or not str(path).lower().endswith((".usd", ".usdc", ".usdz", ".usda")):
            continue
        r = _physics_only_flatten(path, out_dir)
        spawn.usd_path = r["dst"]
        info[name] = r
    return info


def physics_only_nofabric(env_cfg):
    """All headless startup fixes: shared physics material + physics-only .usda assets + no Fabric / no USD sync."""
    info = physics_only_shared_mat(env_cfg)
    _patch_no_fabric_no_usd_sync()
    info["no_fabric_no_usd_sync"] = True
    return info


def physics_only_assets_nofabric(env_cfg, out_dir="/tmp/physics_only_assets"):
    """Task-agnostic headless fixes (e.g. NIST, which has no insertive/receptive objects): every USD-spawned scene asset
    -> physics-only flattened .usda, plus no Fabric / no USD write-back."""
    info = {}
    for name in list(vars(env_cfg.scene)):
        obj = getattr(env_cfg.scene, name, None)
        spawn = getattr(obj, "spawn", None)
        path = getattr(spawn, "usd_path", None)
        if not path or not str(path).lower().endswith((".usd", ".usdc", ".usdz", ".usda")):
            continue
        r = _physics_only_flatten(path, out_dir)
        spawn.usd_path = r["dst"]
        info[name] = {k: v for k, v in r.items() if k != "src"}
    _patch_no_fabric_no_usd_sync()
    info["no_fabric_no_usd_sync"] = True
    return info


def scale_physx_gpu_buffers(env_cfg, ref_envs=32768):
    """Scale the PhysX GPU buffer capacities that a task preset raised for ONE big process (NIST: rigid patches 2^23,
    found/lost pairs 2^22, collision stack 2^32-1 B) to this process's env count, never below PhysxCfg's defaults,
    so several processes fit on one GPU. Overflow would show up as PhysX buffer warnings in the log."""
    from isaaclab_physx.physics import PhysxCfg
    phys = env_cfg.sim.physics
    default = PhysxCfg()
    n = env_cfg.scene.num_envs
    f = n / ref_envs
    info = {}
    for key in ("gpu_max_rigid_contact_count", "gpu_max_rigid_patch_count", "gpu_found_lost_pairs_capacity",
                "gpu_found_lost_aggregate_pairs_capacity", "gpu_total_aggregate_pairs_capacity", "gpu_collision_stack_size",
                "gpu_heap_capacity", "gpu_temp_buffer_capacity"):
        v, d = getattr(phys, key, None), getattr(default, key, None)
        if isinstance(v, int) and isinstance(d, int) and v > d and f < 1:
            nv = max(d, min(v, 1 << int(v * f * 1.25).bit_length()))
            setattr(phys, key, nv)
            info[key] = (v, nv)
    return info


def physics_only_nofabric_scaled(env_cfg, ref_envs=65536):
    """physics_only_nofabric + the task's PhysX GPU buffers scaled to this simulator's envs, for several simulators per
    GPU. The task sizes these buffers for ONE big simulator (collision stack 2 GiB, patch count 2^23, aggregate pairs
    2^23) and every simulator on the GPU reserves them in full. ref_envs = 65,536: one simulator of that size runs the
    expert with the full buffers. 4 x 16,384 envs per A100: peak GPU memory 76 -> 58 GiB, same success, no overflow
    warnings. A heavier scene may need a smaller ref_envs (larger buffers); PhysX logs a warning when one overflows."""
    info = physics_only_nofabric(env_cfg)
    info["buffers"] = scale_physx_gpu_buffers(env_cfg, ref_envs=ref_envs)
    return info


def nist_multiprocess(env_cfg):
    """NIST, several PhysX processes per GPU: physics-only assets + no Fabric + PhysX GPU buffers scaled to envs."""
    info = physics_only_assets_nofabric(env_cfg)
    info["buffers"] = scale_physx_gpu_buffers(env_cfg)
    return info


def nist_state_table_4096(env_cfg, size=4096):
    """NIST: smaller reset-state table per process (default 32,768). The difficulty curriculum promotes when the success
    rate averaged over the WHOLE table passes 0.8; with 32,768 states and 4,096 envs per process most states are rarely
    visited, so that average stays low and difficulty never leaves 0."""
    params = env_cfg.events.reset_strategies.params
    old = params.get("state_table_size")
    params["state_table_size"] = size
    return dict(state_table_size=(old, size))
