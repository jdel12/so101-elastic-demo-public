# Deploying the Elastic Stack

This guide sets up Elasticsearch, Kibana, and all the data infrastructure
the system depends on. Everything runs on Kubernetes via the ECK operator.

## ECK operator

Elastic Cloud on Kubernetes (ECK) manages the Elasticsearch and Kibana
lifecycle — deployment, upgrades, TLS certificates, user secrets. Install
the operator first:

```bash
sudo kubectl create -f https://download.elastic.co/downloads/eck/3.0.0/crds.yaml
sudo kubectl apply -f https://download.elastic.co/downloads/eck/3.0.0/operator.yaml
```

Wait for the operator pod to be ready:

```bash
sudo kubectl get pods -n elastic-system
```

The ECK operator runs in its own namespace and watches for Elasticsearch
and Kibana custom resources across all namespaces.

## Elasticsearch

Deploy Elasticsearch using the manifest in this repo:

```bash
sudo kubectl apply -f deploy/infrastructure/elasticsearch.yaml
```

This creates a 2-node Elasticsearch cluster with 50Gi persistent storage
using k3s's `local-path` provisioner. Wait for it to go green:

```bash
sudo kubectl get elasticsearch
# Wait for HEALTH to show "green" and PHASE to show "Ready"
```

This takes 2-3 minutes on first deploy.

### Get the password

ECK auto-generates the `elastic` superuser password:

```bash
sudo kubectl get secret elasticsearch-es-elastic-user \
  -o jsonpath='{.data.elastic}' | base64 -d
```

Save this password. You'll need it for the rebuild scripts, the MCP server,
and Refinery. Store it in `deploy/scripts/rebuild/config.env`.

### Expose via NodePort

The manifests expose Elasticsearch on port 31920 and Kibana on port 31561
via NodePort services. These ports are accessible from any machine on your
LAN.

Verify Elasticsearch is reachable:

```bash
curl -sk -u elastic:<password> https://<gpu-server-ip>:31920
```

You should get back a JSON response with the cluster name and version.

## Kibana

Deploy Kibana:

```bash
sudo kubectl apply -f deploy/infrastructure/kibana.yaml
```

Wait for it to become ready:

```bash
sudo kubectl get kibana
```

Verify in a browser: `https://<gpu-server-ip>:31561` (accept the
self-signed certificate). Log in with `elastic` and the password from above.

### Response timeout

Agent Builder tool calls can take 30+ seconds (inference, GPU profiling,
evaluation). The default Kibana timeout will cut them off. The manifest
already includes this setting, but verify it's present:

```yaml
spec:
  config:
    xpack.actions.responseTimeout: "120s"
```

If you're using your own Kibana deployment, add this to your `kibana.yml`.

## Trial license

Several features (Agent Builder, Workflows, Fleet) require at minimum a
trial license. Start a 30-day trial:

```bash
curl -sk -u elastic:<password> \
  -X POST "https://<gpu-server-ip>:31920/_license/start_trial?acknowledge=true"
```

The trial is tied to the cluster UUID. If you delete and recreate the
Elasticsearch cluster, you get a fresh 30-day window.

## Index templates

Apply the index templates that define the schema for all robot data:

```bash
cd deploy/scripts/rebuild
cp config.env.example config.env
# Edit config.env: set ES_URL, KB_URL, ES_PASSWORD

./00-preflight.sh
./01-start-trial.sh
./02-apply-index-templates.sh
```

This creates templates for:

| Template | Purpose |
|----------|---------|
| `robot-inference` | Trial results — success/failure, confidence, timing |
| `robot-steps` | Per-step model decisions and joint commands |
| `robot-episodes` | Training episode metadata |
| `robot-commands` | MCP command history |
| `robot-training-buffer` | Hard examples for retraining |
| `robot-training-runs` | Training run metadata |
| `robot-training-coverage` | Embedding space with failure labels |
| `robot-models` | Model checkpoint metadata |
| `robot-evaluations` | VLM evaluation verdicts |
| `robot-scene-embeddings` | CLIP embeddings (1024-dim vectors) |

### Create the robot namespace

The robot workloads run in a dedicated namespace:

```bash
sudo kubectl create namespace robot
```

The Elasticsearch password secret needs to be available in this namespace:

```bash
sudo kubectl get secret elasticsearch-es-elastic-user -o yaml \
  | sed 's/namespace: default/namespace: robot/' \
  | sudo kubectl apply -f -
```

## Fleet Server

Fleet Server manages Elastic Agents across both nodes. It handles agent
enrollment, policy distribution, and integration management.

```bash
sudo kubectl apply -f deploy/infrastructure/fleet.yaml
```

After Fleet Server is running, you can enroll Elastic Agents on both nodes
through the Fleet UI in Kibana (Management > Fleet).

## What you should have at the end

- [ ] ECK operator running in `elastic-system` namespace
- [ ] Elasticsearch cluster green with 2 nodes
- [ ] Kibana accessible at `https://<gpu-server>:31561`
- [ ] Trial license active (30-day window)
- [ ] All 10 index templates applied
- [ ] `robot` namespace created with ES password secret
- [ ] Fleet Server running
- [ ] Password saved in `config.env`

Next: [02-robot-stack.md](02-robot-stack.md) — building and deploying the
robot containers.
