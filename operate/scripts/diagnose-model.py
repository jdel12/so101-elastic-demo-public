#!/usr/bin/env python3
"""Model diagnostic suite — reproduction test, normalization audit, training data profile.

Answers the question: is the failure in the model weights, the pipeline, or the data?

Usage (on cortex, inside lerobot env):
    python3 scripts/diagnose-model.py /path/to/pretrained_model
    python3 scripts/diagnose-model.py /path/to/pretrained_model --episodes 0,1,2 --frames 5
    python3 scripts/diagnose-model.py --compare /path/to/model_a /path/to/model_b
    python3 scripts/diagnose-model.py --profile-data  # training data analysis only
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Prevent broken Pi0 import from blocking XVLA/SmolVLA loading.
# The Pi0 modeling_pi0.py has a syntax error from a training monkey-patch.
# We preload a stub so lerobot.policies.__init__ doesn't choke.
import types
import sys

_fake_pi0 = types.ModuleType("lerobot.policies.pi0")
_fake_pi0.__path__ = []
_fake_pi0.__file__ = "<stub>"

_fake_pi0_config = types.ModuleType("lerobot.policies.pi0.configuration_pi0")
_fake_pi0_config.PI0Config = type("PI0Config", (), {})

_fake_pi0_modeling = types.ModuleType("lerobot.policies.pi0.modeling_pi0")
_fake_pi0_modeling.PI0Policy = type("PI0Policy", (), {})

_fake_pi0.PI0Config = _fake_pi0_config.PI0Config
_fake_pi0.PI0Policy = _fake_pi0_modeling.PI0Policy

sys.modules["lerobot.policies.pi0"] = _fake_pi0
sys.modules["lerobot.policies.pi0.configuration_pi0"] = _fake_pi0_config
sys.modules["lerobot.policies.pi0.modeling_pi0"] = _fake_pi0_modeling


DATASET_ID = "local/blue_cube_2cam_v2_v3.0"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Our deployment environment
DEPLOY_ENV = {
    "cameras": ["observation.images.wrist", "observation.images.overhead"],
    "n_cameras": 2,
    "action_dim": 6,
    "fps": 30,
    "max_reasonable_chunk": 30,
}

# Known failure patterns from this project (add new ones as discovered)
KNOWN_FAILURES = {
    "smolvla_empty_cameras": "SmolVLA v1-v7: empty_cameras=0 told model to expect 3 cameras, we provide 2. 7/7 failure.",
    "xvla_v4_double_count": "X-VLA V4/V4b: fine-tuning from V1 with empty_cameras=1 double-counted (config already had empty_camera_0).",
    "v5v6_action_dim": "V5/V6 first attempt: used training wrapper for pretrained base. action_dim stayed 20 instead of adapting to 6.",
    "fps_mismatch": "Inference at 10 FPS vs training at 30 FPS caused mode collapse.",
    "dataset_version": "LeRobot v3.0 code can't load v2.1 dataset. Must convert first.",
    "chunk_size_50": "SmolVLA default chunk_size=50 causes stutter — 50 steps at 30 FPS = 1.67s between forward passes.",
}


def config_preflight(checkpoint_path, verbose=True):
    """Pre-deployment config audit. Catches known failure patterns before wasting a session.

    Every bug that burned us becomes a check here. Add new checks as new failures are discovered.
    """
    results = []
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path / "config.json"

    if not config_path.exists():
        return [{"check": "config_exists", "status": "FAIL", "detail": f"No config.json at {checkpoint_path}"}]

    cfg = json.load(open(config_path))
    policy_type = cfg.get("type", cfg.get("_target_", "")).lower()

    if verbose:
        print(f"\n{'='*70}")
        print(f"CONFIG PREFLIGHT: {checkpoint_path}")
        print(f"Policy type: {policy_type}")
        print(f"{'='*70}")

    # --- Check 1: Camera/image feature count vs empty_cameras ---
    input_features = cfg.get("input_features", {})
    image_features = [k for k in input_features if "image" in k.lower()]
    empty_cameras = cfg.get("empty_cameras", 0)
    n_image_views = cfg.get("num_image_views", len(image_features))

    real_camera_slots = n_image_views - empty_cameras
    expected_cameras = DEPLOY_ENV["n_cameras"]

    if real_camera_slots != expected_cameras:
        status = "FAIL"
        detail = (f"Model expects {real_camera_slots} real cameras "
                  f"(num_image_views={n_image_views} - empty_cameras={empty_cameras}), "
                  f"but deployment has {expected_cameras}. "
                  f"This mismatch caused SmolVLA v1-v7 failures (7/7).")
    else:
        status = "PASS"
        detail = f"Camera count matches: {real_camera_slots} expected, {expected_cameras} available"
    results.append({"check": "camera_count", "status": status, "detail": detail})

    # --- Check 1b: empty_cameras specifically for SmolVLA ---
    if "smolvla" in policy_type or "smolvla" in str(checkpoint_path).lower():
        base_expects = len(image_features) if image_features else 3
        needed_empty = base_expects - expected_cameras
        if empty_cameras != needed_empty and needed_empty >= 0:
            status = "WARN"
            detail = (f"SmolVLA base has {base_expects} image slots, we have {expected_cameras} cameras. "
                      f"empty_cameras should be {needed_empty}, got {empty_cameras}.")
        else:
            status = "PASS"
            detail = f"empty_cameras={empty_cameras} correct for {expected_cameras} cameras with {base_expects} slots"
        results.append({"check": "smolvla_empty_cameras", "status": status, "detail": detail})

    # --- Check 2: Action dimension ---
    action_dim = cfg.get("output_action_dim", cfg.get("action_dim", None))
    output_features = cfg.get("output_features", {})
    if action_dim and action_dim != DEPLOY_ENV["action_dim"]:
        if action_dim == 20:
            detail = (f"action_dim={action_dim} (pretrained base default). "
                      f"Dataset has {DEPLOY_ENV['action_dim']}. "
                      f"Use lerobot-train directly (not wrapper) to auto-adapt.")
        else:
            detail = f"action_dim={action_dim}, deployment expects {DEPLOY_ENV['action_dim']}"
        results.append({"check": "action_dim", "status": "FAIL", "detail": detail})
    elif action_dim:
        results.append({"check": "action_dim", "status": "PASS",
                        "detail": f"action_dim={action_dim} matches deployment"})

    # --- Check 3: Chunk size ---
    chunk_size = cfg.get("chunk_size", None)
    n_action_steps = cfg.get("n_action_steps", chunk_size)
    if chunk_size:
        if chunk_size > DEPLOY_ENV["max_reasonable_chunk"]:
            detail = (f"chunk_size={chunk_size} means {chunk_size/30:.1f}s between forward passes at 30 FPS. "
                      f"SmolVLA default is 50 — caused stutter. Recommend ≤{DEPLOY_ENV['max_reasonable_chunk']}.")
            results.append({"check": "chunk_size", "status": "WARN", "detail": detail})
        else:
            results.append({"check": "chunk_size", "status": "PASS",
                           "detail": f"chunk_size={chunk_size} (n_action_steps={n_action_steps})"})

    # --- Check 4: Fine-tune double-counting ---
    if "xvla" in policy_type or "xvla" in str(checkpoint_path).lower():
        empty_cam_features = [k for k in input_features if "empty_camera" in k.lower()]
        if empty_cam_features and empty_cameras > 0:
            detail = (f"Config has {len(empty_cam_features)} empty_camera features AND empty_cameras={empty_cameras}. "
                      f"This double-counts — set empty_cameras=0 when fine-tuning from a checkpoint that "
                      f"already has empty cameras baked into input_features. Killed X-VLA V4/V4b.")
            results.append({"check": "double_count", "status": "FAIL", "detail": detail})
        elif empty_cam_features:
            results.append({"check": "double_count", "status": "PASS",
                           "detail": f"empty_cameras=0 with {len(empty_cam_features)} baked empty features — correct for fine-tune"})

    # --- Check 5: Preprocessor/postprocessor files ---
    pre_path = checkpoint_path / "policy_preprocessor.json"
    post_path = checkpoint_path / "policy_postprocessor.json"
    missing = []
    if not pre_path.exists():
        missing.append("policy_preprocessor.json")
    if not post_path.exists():
        missing.append("policy_postprocessor.json")
    if missing:
        results.append({"check": "processor_files", "status": "FAIL",
                        "detail": f"Missing: {', '.join(missing)}. Model can't normalize/denormalize."})
    else:
        results.append({"check": "processor_files", "status": "PASS",
                        "detail": "Preprocessor and postprocessor configs present"})

    # --- Check 6: Normalization stats ---
    unnorm_files = list(checkpoint_path.glob("*unnormalizer*"))
    if not unnorm_files:
        results.append({"check": "norm_stats", "status": "WARN",
                        "detail": "No unnormalizer safetensors found. Output may be in normalized space."})
    else:
        results.append({"check": "norm_stats", "status": "PASS",
                        "detail": f"Found {len(unnorm_files)} unnormalizer file(s)"})

    # --- Check 7: Model weights ---
    safetensor_files = list(checkpoint_path.glob("*.safetensors"))
    weight_files = [f for f in safetensor_files if "unnormalizer" not in f.name]
    if not weight_files:
        pt_files = list(checkpoint_path.glob("*.bin")) + list(checkpoint_path.glob("*.pt"))
        if pt_files:
            results.append({"check": "weights", "status": "WARN",
                           "detail": f"PyTorch format ({len(pt_files)} files), not safetensors. May need conversion."})
        else:
            results.append({"check": "weights", "status": "FAIL",
                           "detail": "No model weight files found"})
    else:
        total_mb = sum(f.stat().st_size for f in weight_files) / 1e6
        results.append({"check": "weights", "status": "PASS",
                       "detail": f"{len(weight_files)} safetensors file(s), {total_mb:.0f} MB total"})

    # --- Check 8: Input feature naming vs our cameras ---
    our_cam_names = set(DEPLOY_ENV["cameras"])
    config_cam_names = set()
    for feat_name in input_features:
        if "image" in feat_name.lower() and "empty" not in feat_name.lower():
            config_cam_names.add(feat_name)

    if config_cam_names and not config_cam_names.intersection(our_cam_names):
        detail = (f"Model expects cameras: {sorted(config_cam_names)}, "
                  f"we provide: {sorted(our_cam_names)}. "
                  f"Need --rename_map in training or camera topic remapping at inference.")
        results.append({"check": "camera_names", "status": "WARN", "detail": detail})
    elif config_cam_names:
        matched = config_cam_names.intersection(our_cam_names)
        results.append({"check": "camera_names", "status": "PASS",
                        "detail": f"Camera names match: {sorted(matched)}"})

    # --- Check 9: Symlinks (broken symlinks are a common deployment issue) ---
    broken_links = [f for f in checkpoint_path.iterdir()
                    if f.is_symlink() and not f.resolve().exists()]
    if broken_links:
        results.append({"check": "broken_symlinks", "status": "FAIL",
                        "detail": f"{len(broken_links)} broken symlinks: {[f.name for f in broken_links]}"})

    # --- Print results ---
    if verbose:
        n_pass = sum(1 for r in results if r["status"] == "PASS")
        n_warn = sum(1 for r in results if r["status"] == "WARN")
        n_fail = sum(1 for r in results if r["status"] == "FAIL")

        for r in results:
            icon = {"PASS": " OK ", "WARN": "WARN", "FAIL": "FAIL"}[r["status"]]
            print(f"  [{icon}] {r['check']:25s} {r['detail']}")

        print(f"\n  {'='*50}")
        print(f"  PREFLIGHT: {n_pass} passed, {n_warn} warnings, {n_fail} failures")

        if n_fail > 0:
            print(f"  VERDICT: DO NOT DEPLOY — {n_fail} blocking issue(s)")
            print(f"\n  Known failure patterns that match:")
            for key, desc in KNOWN_FAILURES.items():
                for r in results:
                    if r["status"] == "FAIL" and key.replace("_", " ") in r["detail"].lower():
                        print(f"    - {desc}")
                        break
        elif n_warn > 0:
            print(f"  VERDICT: DEPLOY WITH CAUTION — {n_warn} warning(s)")
        else:
            print(f"  VERDICT: CLEAR FOR DEPLOYMENT")

    return results


def load_model(checkpoint_path, device="cuda"):
    """Load any LeRobot policy + pre/post processors."""
    cfg = json.load(open(f"{checkpoint_path}/config.json"))
    policy_type = cfg.get("type", cfg.get("_target_", "")).lower()

    # Bypass lerobot.policies.__init__ which imports ALL policies (Pi0 may have syntax errors from monkey-patches)
    import importlib, importlib.util
    def _load_policy_module(subpath):
        spec = importlib.util.find_spec(subpath)
        if spec is None:
            raise ImportError(f"Cannot find {subpath}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    if "xvla" in policy_type or "xvla" in str(checkpoint_path).lower():
        mod = _load_policy_module("lerobot.policies.xvla.modeling_xvla")
        policy = mod.XVLAPolicy.from_pretrained(checkpoint_path)
        ptype = "xvla"
    elif "pi0" in policy_type:
        mod = _load_policy_module("lerobot.policies.pi0.modeling_pi0")
        policy = mod.PI0Policy.from_pretrained(checkpoint_path)
        ptype = "pi0"
    elif "smolvla" in policy_type or "smolvla" in str(checkpoint_path).lower():
        mod = _load_policy_module("lerobot.policies.smolvla.modeling_smolvla")
        policy = mod.SmolVLAPolicy.from_pretrained(checkpoint_path)
        ptype = "smolvla"
    else:
        mod = _load_policy_module("lerobot.policies.xvla.modeling_xvla")
        policy = mod.XVLAPolicy.from_pretrained(checkpoint_path)
        ptype = "xvla"

    dev = torch.device(device)
    policy.to(dev)
    policy.eval()

    try:
        from lerobot.policies.factory import make_pre_post_processors
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config, checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": dev}},
        )
    except (ImportError, Exception):
        from lerobot.processor.pipeline import DataProcessorPipeline
        preprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_path, "policy_preprocessor.json"
        )
        postprocessor = DataProcessorPipeline.from_pretrained(
            checkpoint_path, "policy_postprocessor.json"
        )

    return policy, preprocessor, postprocessor, ptype, dev


def load_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(DATASET_ID)


def get_normalization_stats(checkpoint_path):
    """Extract normalization stats from a checkpoint's postprocessor."""
    post_cfg_path = Path(checkpoint_path) / "policy_postprocessor.json"
    if not post_cfg_path.exists():
        return None

    post_cfg = json.load(open(post_cfg_path))

    stats_path = Path(checkpoint_path) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if stats_path.exists():
        from safetensors.torch import load_file
        tensors = load_file(str(stats_path))
        return {k: v.numpy() for k, v in tensors.items()}
    return None


def reproduction_test(checkpoint_path, episode_indices, n_frames_per_ep, device="cuda"):
    """Feed real training frames through the model, compare output to training labels.

    If the model can't reproduce its own training data, the bug is in the pipeline
    (normalization, postprocessor, camera preprocessing), not the weights.
    """
    print(f"\n{'='*70}")
    print(f"REPRODUCTION TEST: {checkpoint_path}")
    print(f"Episodes: {episode_indices}, frames per episode: {n_frames_per_ep}")
    print(f"{'='*70}")

    policy, preprocessor, postprocessor, ptype, dev = load_model(checkpoint_path, device)
    ds = load_dataset()

    # Reset policy internal state
    if hasattr(policy, '_queues'):
        from collections import deque
        policy._queues = {k: deque(maxlen=v.maxlen) for k, v in policy._queues.items()}

    all_errors = []
    per_joint_errors = [[] for _ in range(6)]
    per_episode_results = []

    for ep_idx in episode_indices:
        ep_data = ds.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        n_frames = len(ep_data)

        if n_frames == 0:
            print(f"  Episode {ep_idx}: no data, skipping")
            continue

        # Sample frames evenly across the episode
        if n_frames_per_ep >= n_frames:
            frame_indices = list(range(n_frames))
        else:
            frame_indices = np.linspace(0, n_frames - 1, n_frames_per_ep, dtype=int).tolist()

        ep_errors = []
        ep_per_joint = [[] for _ in range(6)]

        print(f"\n  Episode {ep_idx} ({n_frames} frames, sampling {len(frame_indices)}):")

        for fi in frame_indices:
            # Get the actual dataset sample (handles video decoding)
            global_idx = int(ep_data[fi]["index"])
            sample = ds[global_idx]

            # Build observation
            obs = {
                "observation.state": sample["observation.state"],
                "task": "Pick up the blue cube and place it in the bin",
            }
            for cam_key in ["observation.images.wrist", "observation.images.overhead"]:
                if cam_key in sample:
                    obs[cam_key] = sample[cam_key]

            # Ground truth action (already in joint degrees)
            gt_action = sample["action"].numpy()

            # Run full pipeline
            if hasattr(policy, '_queues'):
                from collections import deque
                policy._queues = {k: deque(maxlen=v.maxlen) for k, v in policy._queues.items()}

            batch = preprocessor(obs)

            with torch.no_grad():
                action = policy.select_action(batch)

            action_in = action.unsqueeze(0) if action.dim() == 1 else action
            output = postprocessor(action_in)
            if isinstance(output, dict):
                predicted = output["action"].squeeze().cpu().numpy()[:6]
            else:
                predicted = output.squeeze().cpu().numpy()[:6]

            error = predicted - gt_action
            abs_error = np.abs(error)

            all_errors.append(abs_error)
            ep_errors.append(abs_error)
            for j in range(6):
                per_joint_errors[j].append(abs_error[j])
                ep_per_joint[j].append(abs_error[j])

            if fi in [frame_indices[0], frame_indices[-1]]:
                print(f"    Frame {fi:4d}: gt=[{', '.join(f'{v:7.1f}' for v in gt_action)}]")
                print(f"              pred=[{', '.join(f'{v:7.1f}' for v in predicted)}]")
                print(f"              err =[{', '.join(f'{v:7.1f}' for v in error)}]")

        ep_mean = np.mean(ep_errors, axis=0)
        per_episode_results.append({
            "episode": ep_idx,
            "n_frames": len(frame_indices),
            "mean_abs_error": ep_mean.tolist(),
            "mean_total_error": float(np.mean(ep_mean)),
        })
        print(f"    Mean abs error: [{', '.join(f'{v:.1f}' for v in ep_mean)}]  total: {np.mean(ep_mean):.1f}°")

    if not all_errors:
        print("  No data processed!")
        return {}

    all_errors = np.array(all_errors)
    mean_error = np.mean(all_errors, axis=0)
    max_error = np.max(all_errors, axis=0)

    print(f"\n  {'='*50}")
    print(f"  REPRODUCTION TEST SUMMARY")
    print(f"  {'='*50}")
    print(f"  Per-joint mean absolute error (degrees):")
    for j in range(6):
        severity = "!!!" if mean_error[j] > 20 else "!!" if mean_error[j] > 10 else "!" if mean_error[j] > 5 else ""
        print(f"    {JOINT_NAMES[j]:15s}: {mean_error[j]:6.1f}° mean, {max_error[j]:6.1f}° max  {severity}")

    total_mean = float(np.mean(mean_error))
    print(f"\n  Overall mean error: {total_mean:.1f}°")

    if total_mean < 5.0:
        verdict = "PASS — model reproduces training data through pipeline"
        diagnosis = "pipeline_ok"
    elif total_mean < 15.0:
        worst_joints = [JOINT_NAMES[j] for j in range(6) if mean_error[j] > 10]
        verdict = f"MARGINAL — moderate divergence on {', '.join(worst_joints) or 'multiple joints'}"
        diagnosis = "marginal"
    else:
        worst_joints = [JOINT_NAMES[j] for j in range(6) if mean_error[j] > 15]
        verdict = f"FAIL — model cannot reproduce training data. Suspect: normalization or postprocessor"
        diagnosis = "pipeline_broken"

    print(f"\n  VERDICT: {verdict}")

    return {
        "checkpoint": str(checkpoint_path),
        "policy_type": ptype,
        "diagnosis": diagnosis,
        "mean_error_per_joint": mean_error.tolist(),
        "max_error_per_joint": max_error.tolist(),
        "total_mean_error": total_mean,
        "episodes": per_episode_results,
    }


def normalization_audit(checkpoint_paths):
    """Compare normalization stats across checkpoints and against the dataset."""
    print(f"\n{'='*70}")
    print("NORMALIZATION AUDIT")
    print(f"{'='*70}")

    # Load dataset stats
    ds_stats_path = Path.home() / ".cache/huggingface/lerobot/local/blue_cube_2cam_v2/meta/stats.json"
    ds_stats = json.load(open(ds_stats_path)) if ds_stats_path.exists() else None

    if ds_stats:
        print(f"\n  Dataset action stats ({DATASET_ID}):")
        print(f"    {'Joint':15s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
        for j, name in enumerate(JOINT_NAMES):
            print(f"    {name:15s} {ds_stats['action']['mean'][j]:8.2f} {ds_stats['action']['std'][j]:8.2f} "
                  f"{ds_stats['action']['min'][j]:8.2f} {ds_stats['action']['max'][j]:8.2f}")

    results = {}
    for path in checkpoint_paths:
        path = str(path)
        stats = get_normalization_stats(path)
        if stats is None:
            print(f"\n  {path}: no normalization stats found")
            continue

        print(f"\n  Checkpoint: {path}")
        # Look for unnormalizer parameters (mean/std or min/max)
        for key, tensor in sorted(stats.items()):
            if tensor.size <= 6:
                print(f"    {key}: [{', '.join(f'{v:.3f}' for v in tensor.flatten()[:6])}]")

        # Compare to dataset stats if available
        if ds_stats:
            for key, tensor in stats.items():
                t = tensor.flatten()[:6]
                if "mean" in key and "action" in key.lower():
                    ds_mean = np.array(ds_stats["action"]["mean"])
                    drift = np.abs(t - ds_mean)
                    if np.any(drift > 1.0):
                        drifted = [JOINT_NAMES[j] for j in range(6) if drift[j] > 1.0]
                        print(f"    WARNING: mean drift on {', '.join(drifted)}: {drift}")
                elif "std" in key and "action" in key.lower():
                    ds_std = np.array(ds_stats["action"]["std"])
                    ratio = t / (ds_std + 1e-8)
                    if np.any(np.abs(ratio - 1.0) > 0.2):
                        shifted = [JOINT_NAMES[j] for j in range(6) if abs(ratio[j] - 1.0) > 0.2]
                        print(f"    WARNING: std ratio shift on {', '.join(shifted)}: {ratio}")

        results[path] = {k: v.tolist() for k, v in stats.items()}

    return results


def profile_training_data():
    """Analyze training data characteristics: per-episode action distributions,
    gripper patterns, workspace coverage."""
    print(f"\n{'='*70}")
    print("TRAINING DATA PROFILE")
    print(f"{'='*70}")

    ds = load_dataset()
    meta_path = Path.home() / ".cache/huggingface/lerobot" / DATASET_ID / "meta/info.json"
    meta = json.load(open(meta_path))
    n_eps = meta["total_episodes"]

    ep_profiles = []

    for ep in range(n_eps):
        ep_data = ds.hf_dataset.filter(lambda x: x["episode_index"] == ep)
        states = np.array(ep_data["observation.state"])
        actions = np.array(ep_data["action"])
        n_frames = len(states)

        gripper_actions = actions[:, 5]
        gripper_states = states[:, 5]

        # Gripper transition count (direction changes)
        if len(gripper_actions) > 1:
            diffs = np.diff(gripper_actions)
            signs = np.sign(diffs)
            sign_changes = np.sum(np.abs(np.diff(signs[signs != 0])) > 0) if np.any(signs != 0) else 0
            transitions = int(sign_changes)
        else:
            transitions = 0

        # Detect grip event (gripper < 15° for > 10 frames)
        has_grip = bool(np.any(gripper_states < 15.0))
        min_gripper = float(np.min(gripper_states))

        # Initial position (proxy for starting config)
        initial_state = states[0, :5].tolist()

        # Per-joint action range
        action_ranges = (np.max(actions, axis=0) - np.min(actions, axis=0)).tolist()

        # Trajectory smoothness (jerk = 3rd derivative)
        if n_frames > 3:
            vel = np.diff(states[:, :5], axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            mean_jerk = float(np.mean(np.abs(jerk)))
        else:
            mean_jerk = 0

        profile = {
            "episode": ep,
            "n_frames": n_frames,
            "duration_sec": n_frames / 30.0,
            "gripper_transitions": transitions,
            "has_grip": has_grip,
            "min_gripper": round(min_gripper, 1),
            "initial_state": [round(v, 1) for v in initial_state],
            "action_ranges": [round(v, 1) for v in action_ranges],
            "mean_jerk": round(mean_jerk, 2),
        }
        ep_profiles.append(profile)

    # Summary statistics
    grip_episodes = sum(1 for p in ep_profiles if p["has_grip"])
    mean_transitions = np.mean([p["gripper_transitions"] for p in ep_profiles])
    mean_duration = np.mean([p["duration_sec"] for p in ep_profiles])

    # Workspace coverage — cluster initial positions
    initial_positions = np.array([p["initial_state"] for p in ep_profiles])
    position_std = np.std(initial_positions, axis=0)

    print(f"\n  Dataset: {n_eps} episodes, {meta['total_frames']} frames")
    print(f"  Mean duration: {mean_duration:.1f}s ({mean_duration * 30:.0f} frames)")
    print(f"  Episodes with grip event: {grip_episodes}/{n_eps} ({100*grip_episodes/n_eps:.0f}%)")
    print(f"  Mean gripper transitions per episode: {mean_transitions:.1f}")

    print(f"\n  Initial position diversity (std per joint):")
    for j, name in enumerate(JOINT_NAMES[:5]):
        diversity = "LOW" if position_std[j] < 3 else "OK" if position_std[j] < 10 else "HIGH"
        print(f"    {name:15s}: std={position_std[j]:.1f}° [{diversity}]")

    # Per-joint action range consistency
    all_ranges = np.array([p["action_ranges"] for p in ep_profiles])
    range_mean = np.mean(all_ranges, axis=0)
    range_std = np.std(all_ranges, axis=0)
    range_cv = range_std / (range_mean + 1e-8)

    print(f"\n  Action range consistency (coefficient of variation):")
    for j, name in enumerate(JOINT_NAMES):
        consistency = "CONSISTENT" if range_cv[j] < 0.3 else "VARIABLE" if range_cv[j] < 0.6 else "INCONSISTENT"
        print(f"    {name:15s}: mean={range_mean[j]:5.1f}° ± {range_std[j]:5.1f}° (CV={range_cv[j]:.2f}) [{consistency}]")

    # Identify outlier episodes
    mean_ranges = np.mean(all_ranges[:, :5], axis=1)
    range_z = (mean_ranges - np.mean(mean_ranges)) / (np.std(mean_ranges) + 1e-8)
    outliers = [ep_profiles[i] for i in range(n_eps) if abs(range_z[i]) > 2.0]

    if outliers:
        print(f"\n  Outlier episodes (>2σ from mean range):")
        for o in outliers:
            z = range_z[o["episode"]]
            label = "unusually large" if z > 0 else "unusually small"
            print(f"    Episode {o['episode']}: {label} motions (z={z:.1f}), "
                  f"grip={o['has_grip']}, transitions={o['gripper_transitions']}")

    # Gripper analysis
    no_grip = [p for p in ep_profiles if not p["has_grip"]]
    if no_grip:
        print(f"\n  WARNING: {len(no_grip)} episodes have NO grip event:")
        for p in no_grip[:5]:
            print(f"    Episode {p['episode']}: min_gripper={p['min_gripper']}°, "
                  f"transitions={p['gripper_transitions']}")

    return {
        "n_episodes": n_eps,
        "grip_episodes": grip_episodes,
        "mean_transitions": float(mean_transitions),
        "position_diversity_std": position_std.tolist(),
        "action_range_cv": range_cv.tolist(),
        "outlier_episodes": [o["episode"] for o in outliers],
        "no_grip_episodes": [p["episode"] for p in no_grip],
        "profiles": ep_profiles,
    }


def compare_models(checkpoint_paths, episode_indices, n_frames, device="cuda"):
    """Run reproduction test on multiple models, compare side by side."""
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}")

    results = {}
    for path in checkpoint_paths:
        name = Path(path).parent.name if Path(path).name == "pretrained_model" else Path(path).name
        try:
            r = reproduction_test(path, episode_indices, n_frames, device)
            results[name] = r
        except Exception as e:
            print(f"\n  {name}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"diagnosis": "error", "error": str(e)}

    # Side-by-side comparison
    print(f"\n  {'='*50}")
    print(f"  COMPARISON SUMMARY")
    print(f"  {'='*50}")
    print(f"  {'Model':20s} {'Total':>8s} {'Diag':>20s} | {' '.join(f'{n[:6]:>7s}' for n in JOINT_NAMES)}")
    print(f"  {'-'*20} {'-'*8} {'-'*20}-+-{'-'.join('-'*7 for _ in JOINT_NAMES)}")

    for name, r in results.items():
        if "error" in r:
            print(f"  {name:20s} {'ERROR':>8s} {r.get('error','')[:20]:>20s}")
            continue
        total = r.get("total_mean_error", 0)
        diag = r.get("diagnosis", "?")
        errs = r.get("mean_error_per_joint", [0]*6)
        print(f"  {name:20s} {total:7.1f}° {diag:>20s} | {' '.join(f'{e:7.1f}' for e in errs)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model diagnostic suite")
    parser.add_argument("checkpoint", nargs="?", help="Path to pretrained_model dir")
    parser.add_argument("--episodes", default="0,5,10,30,60",
                        help="Comma-separated episode indices to test")
    parser.add_argument("--frames", type=int, default=10,
                        help="Frames to sample per episode")
    parser.add_argument("--compare", nargs="+",
                        help="Compare multiple checkpoints")
    parser.add_argument("--profile-data", action="store_true",
                        help="Profile training data only (no model needed)")
    parser.add_argument("--norm-audit", nargs="+",
                        help="Normalization audit on checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", help="Save results as JSON")
    args = parser.parse_args()

    results = {}
    episode_indices = [int(e) for e in args.episodes.split(",")]

    if args.profile_data:
        results["training_data"] = profile_training_data()

    elif args.norm_audit:
        results["normalization"] = normalization_audit(args.norm_audit)

    elif args.compare:
        results["comparison"] = compare_models(
            args.compare, episode_indices, args.frames, args.device
        )

    elif args.checkpoint:
        # Full diagnostic: reproduction test + normalization audit + data profile
        results["reproduction"] = reproduction_test(
            args.checkpoint, episode_indices, args.frames, args.device
        )
        results["normalization"] = normalization_audit([args.checkpoint])
        results["training_data"] = profile_training_data()

    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
