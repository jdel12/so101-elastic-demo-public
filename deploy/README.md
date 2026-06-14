# Deploy

Bring up the full software stack: Elasticsearch, Kibana, the robot brain,
cameras, motor control, observability, and Agent Builder. At the end of
this stage, you can talk to your robot through Kibana.

| Guide | What you'll deploy |
|-------|-------------------|
| [01-elastic.md](01-elastic.md) | ECK operator, Elasticsearch, Kibana, index templates, trial license |
| [02-robot-stack.md](02-robot-stack.md) | Container images, robot-brain pod, cameras, motor node |
| [03-observability.md](03-observability.md) | APM, Fleet, DCGM Exporter, ROS 2 monitoring, dashboards |
| [04-agent-builder.md](04-agent-builder.md) | MCP connector, Agent Builder agent, tool registration, first trial |

**Time estimate:** 2-4 hours for the full stack, mostly spent building
container images (the inference image is ~12GB).

Infrastructure manifests are in `infrastructure/`. Dockerfiles and k8s
manifests are in `robot/`. Observability configs and dashboard exports are
in `observability/`. Deployment and rebuild scripts are in `scripts/`.
