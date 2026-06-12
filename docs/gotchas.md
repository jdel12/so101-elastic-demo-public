# Gotchas

Hard-won operational knowledge, organized by category. Each entry is a problem
and its fix.

## USB & Hardware

- **USB device paths shift after reboot or re-enumeration.** Camera devices
  (`/dev/video*`) and serial ports (`/dev/ttyACM*`) are assigned dynamically.
  **Fix:** Check `v4l2-ctl --list-devices` and `ls /dev/ttyACM*` before every
  session. Write udev rules to create stable symlinks, but note that udev
  symlinks do not propagate into k3s containers even when privileged -- use raw
  device paths in k8s manifests.

- **Two USB cameras on the same root hub causes frame drops.** USB 2.0
  bandwidth is shared per controller, and two 1080p streams exceed it.
  **Fix:** Plug cameras into ports on separate USB controllers. Verify with
  `lsusb -t` -- each camera should be on a different bus number.

- **USB serial devices re-enumerate when the arm is power-cycled.** The
  follower arm's serial port may change from `/dev/ttyACM0` to `/dev/ttyACM1`
  or vice versa. **Fix:** Check `ls /dev/ttyACM*` on the edge node after any
  power cycle and update the `FOLLOWER_PORT` env var in the manifest.

## Kubernetes

- **hostNetwork pods block rolling updates.** When a pod uses `hostNetwork:
  true` and binds a host port, the replacement pod cannot start until the old
  one releases the port. **Fix:** Delete the old pod before applying changes:
  `kubectl delete pod -n robot -l app=<name>` then `kubectl apply`.

- **Secrets are namespace-scoped.** The ECK-generated `elasticsearch-es-elastic-user`
  secret lives in the `default` namespace. Pods in the `robot` namespace cannot
  access it. **Fix:** Copy the secret to the robot namespace:
  `kubectl get secret ... -o yaml | sed 's/namespace: default/namespace: robot/' | kubectl apply -f -`.

- **Trial license is one per cluster UUID.** You cannot restart a trial on the
  same cluster. **Fix:** To get a new trial, delete and recreate the
  Elasticsearch custom resource. This generates a new cluster UUID.

- **k3s uses containerd, not Docker.** `docker build` images are not visible to
  k3s. **Fix:** Push images to a registry (local or remote) and reference them
  in manifests with the registry URL.

## MCP & Agent Builder

- **Agent Builder tool IDs use a connector prefix.** Tools must be registered
  as `robot.<tool_name>`, not just `<tool_name>`. **Fix:** Use the full
  prefixed ID in the registration API call.

- **The `execute_task` parameter is `task`, not `task_instruction`.** The LLM
  sometimes hallucinates the wrong parameter name. **Fix:** Ensure the agent
  system prompt specifies the correct parameter name, or handle both in the
  tool implementation.

- **FastMCP SSE transport needs proper headers.** The Kibana MCP connector
  communicates over SSE (Server-Sent Events). Requests need the `Accept:
  text/event-stream` header, and the client must parse the SSE event stream
  format. **Fix:** Use the `callTool` subAction through the connector API
  (not `toolsCall` or `run`), which handles SSE internally.

- **Kibana cuts off long-running tool calls.** The default action timeout is
  30 seconds. Inference and evaluation tools can take 30-60 seconds.
  **Fix:** Set `xpack.actions.responseTimeout: "120s"` in Kibana config.

- **Connector ID changes after cluster rebuild.** The MCP connector is assigned
  a new UUID when re-imported. **Fix:** After running the saved object import
  script, note the new connector ID and update any hardcoded references.

## VLA Models

- **INFERENCE_FPS must match the training dataset FPS.** If the model was
  trained on 30 FPS data, inference must run at 30 FPS. Running at 10 FPS
  produces slow, distorted motions. **Fix:** Check the dataset metadata for
  FPS and set `INFERENCE_FPS` to match.

- **Model swap path must be the container path, not the host path.** The
  `swap_model` tool operates inside the container. A host path like
  `/home/user/models/v2` does not exist in the container. **Fix:** Use the
  container mount path (e.g., `/checkpoints/xvla_v2`).

- **`empty_cameras` must match the training configuration.** If training used
  `empty_cameras=0` (all cameras active), inference must use the same setting.
  Mismatches cause shape errors or silent degradation. **Fix:** Check the
  training config and match it exactly.

- **LeRobot X-VLA integration has known bugs.** As of LeRobot 0.4.3, the
  X-VLA code path has bugs in `rtc_config`, `vlm_model_name`, `detect`,
  `extras`, and symlink resolution. **Fix:** Apply the monkeypatch in
  `inference_node.py` which patches these at import time before model loading.

- **Fine-tuning from a fine-tuned checkpoint causes catastrophic forgetting.**
  Training a model from another fine-tuned checkpoint (not the pretrained base)
  produces models that hover or exhibit gripper mode collapse. **Fix:** Always
  fine-tune from the pretrained base model, not from a previous fine-tune.

## ROS 2

- **FastDDS SHM breaks when mixing ROS 2 versions.** Humble containers use
  FastDDS 6.x and Jazzy containers use FastDDS 8.x. Shared memory transport
  between different major versions silently fails. **Fix:** Disable SHM by
  setting `FASTRTPS_DEFAULT_PROFILES_FILE` to a config that forces UDP-only
  transport (see `infra/docker/fastdds_no_shm.xml`).

- **Source the correct `setup.bash` for your ROS 2 version.** Humble and Jazzy
  have different setup paths (`/opt/ros/humble/setup.bash` vs
  `/opt/ros/jazzy/setup.bash`). Sourcing the wrong one produces cryptic import
  errors. **Fix:** Container entrypoints source the correct file automatically.
  On the host, use the version that matches your target.

- **Motor type mismatch is silent.** If the motor driver expects STS3215 servos
  but the arm has a different servo type, commands are sent but produce no
  motion or wrong motion with no error message. **Fix:** Verify the servo type
  in the LeRobot configuration matches the physical hardware.

- **DDS multicast requires a wired network.** Wi-Fi networks often block or
  throttle multicast traffic. DDS discovery fails silently. **Fix:** Use
  gigabit Ethernet between all ROS 2 nodes.

## Elasticsearch

- **Kibana is HTTPS-only under ECK.** ECK generates self-signed TLS
  certificates. All Kibana API calls require `https://`, and curl needs `-k`
  to skip certificate verification. **Fix:** Use `curl -sk` or install the CA
  certificate.

- **Workflows require a trial license.** The Elastic Workflows feature is not
  available on a basic license. **Fix:** Start a trial license before deploying
  workflows.

- **Connector secrets are not exported with saved objects.** When you import
  saved objects (connectors, dashboards), connector secrets (API keys, URLs)
  are stripped. **Fix:** After import, open each connector in the Kibana UI
  and re-enter the secret values.

- **Index templates must exist before the first document.** If data arrives
  before the template is applied, the index gets default mappings that may not
  match your queries. **Fix:** Run `02-apply-index-templates.sh` before
  starting any robot workloads.

## CLIP & Evaluation

- **Embed the scene before execution, not after.** Post-execution frames show
  the robot arm in the workspace, which changes the CLIP distance measurement.
  **Fix:** Capture and embed the initial scene frame before calling
  `execute_task`.

- **The overhead camera is too coarse for fine position estimation.** CLIP
  distance from overhead frames detects gross displacement but misses small
  positional shifts. **Fix:** Use the wrist camera for fine-grained novelty
  detection and the overhead camera for coarse scene verification.

- **Qwen3-VL sometimes returns malformed JSON.** The VLM evaluator
  occasionally wraps JSON in markdown code fences or adds commentary.
  **Fix:** The MCP server includes JSON extraction logic that strips fences
  and retries. If evaluation fails, check the raw VLM response in the trial
  document.

- **Do not touch objects during evaluation.** Moving the cube or other objects
  between execution and evaluation invalidates the VLM and CLIP assessments.
  **Fix:** Wait for the full `evaluate_cycle` to complete before resetting the
  workspace.

## GPU

- **Consumer GPUs block CUPTI profiling by default.** NVIDIA restricts CUPTI
  access on GeForce cards. **Fix:** Set
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility` and run the container in
  privileged mode. Some driver versions require additional bypass configuration.

- **The CUPTI shim path must match the on-disk inode.** If Refinery is
  reinstalled, `libgpucollector.so` gets a new inode. The inference container
  caches the old file descriptor. **Fix:** Restart the inference container
  after any Refinery update.

- **ES 9.x defaults to `bbq_hnsw` for vector indices.** Some GPU-accelerated
  vector search backends (e.g., cuVS) are incompatible with `bbq_hnsw`.
  **Fix:** Explicitly set the index type to `hnsw` in the index template if
  using GPU-accelerated search.

## Training

- **Tilde (`~`) is not expanded in Python subprocess calls.** Paths like
  `~/outputs/checkpoint` are passed literally, not expanded to the home
  directory. **Fix:** Use `$HOME` in shell scripts or `os.path.expanduser()`
  in Python. This can silently create directories under a literal `~` folder.

- **Docker build context is the repository root.** Dockerfiles use `COPY`
  paths relative to the repo root, not the Dockerfile location. **Fix:** Always
  run `docker build` from the repository root with `-f infra/docker/Dockerfile.X`.

- **Base model vs. checkpoint training use different code paths.** Training
  from a pretrained base model and from a fine-tuned checkpoint use different
  configuration parameters in LeRobot. **Fix:** Use the correct training
  script for your starting point. The wrapper script is checkpoint-only;
  training from the base model requires `lerobot-train` directly.

- **Pretrained X-VLA has 20-dimensional action space.** The base model expects
  a different action dimension than the fine-tuned checkpoints. **Fix:** Let
  LeRobot handle the action space mapping automatically via the dataset
  configuration.

## Docker & Containers

- **Host-mounting source files eliminates the rebuild cycle.** The MCP server
  and inference node are mounted from the host via k8s `hostPath` volumes.
  Edits on the host take effect on pod restart. **Fix:** Only rebuild images
  when Python dependencies change. For code changes, just restart the pod.

- **Conda environments override system Python in ROS 2 containers.** If a
  conda environment is active, `python3` resolves to the conda Python, which
  does not have ROS 2 packages. **Fix:** Deactivate conda before building or
  entering ROS 2 containers:
  `conda deactivate` or unset `CONDA_DEFAULT_ENV`.

- **Container registry must be trusted by k3s on all nodes.** If using an
  HTTP (not HTTPS) registry, both the GPU server and edge node need the
  registry configured in `/etc/rancher/k3s/registries.yaml`. **Fix:** Add the
  registry mirror configuration on every k3s node and restart k3s.
