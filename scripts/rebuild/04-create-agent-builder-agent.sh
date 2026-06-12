#!/usr/bin/env bash
# 04-create-agent-builder-agent.sh — create the SO-101 Diagnostic Agent
# in Kibana Agent Builder via /api/agent_builder/agents.
#
# Prerequisites:
# - Trial license or Platinum/Enterprise (feature is not available on Basic)
# - robot-arm MCP connector already imported (script 03)
# - Ollama Local connector configured with a valid Ollama endpoint
#
# IMPORTANT — Two-step gotcha for MCP tools in Agent Builder:
#
#   Agent Builder does NOT automatically discover tools from an MCP connector.
#   You must explicitly register each tool via POST /api/agent_builder/tools
#   BEFORE referencing it in the agent's tool list. The tool IDs in the agent
#   must use connector-prefixed format: "robot.<tool_name>" (not bare names).
#
#   Step 1: POST /api/agent_builder/tools for each tool with:
#           {"id": "robot.<tool_name>", "type": "mcp",
#            "configuration": {"connector_id": "robot-arm", "tool_name": "<tool_name>"}}
#
#   Step 2: POST /api/agent_builder/agents with tool_ids referencing "robot.*" IDs.
#
#   If you skip Step 1, the agent creation will succeed but the tools won't resolve
#   at inference time. This is a known Agent Builder UX gap.
#
# The agent system prompt below is derived from the MCP server's self-reported
# "instructions" field pulled from the live MCP init response.
# Feel free to customize before running.

source "$(dirname "$0")/lib.sh"

AGENT_NAME="SO-101 Diagnostic Agent"
AGENT_ID="so101-diagnostic-agent"
MCP_CONNECTOR_ID="robot-arm"

# The 26 MCP tools exposed by mcp_server.py v15
MCP_TOOLS=(
  # Robot control
  execute_task
  stop_robot
  swap_model
  get_robot_status
  # Observation
  get_workspace_snapshot
  evaluate_cycle
  # Conditions
  set_condition
  get_condition
  # Data & stats
  get_flywheel_status
  query_hard_examples
  query_inference_stats
  # GPU observability (Refinery)
  get_gpu_kernel_profile
  get_inference_breakdown
  get_gpu_memory_status
  get_episode_gpu_quality
  get_control_loop_status
  get_safety_events
  # Diagnostic
  diagnose_trial
  compare_model_kernels
  check_scene_novelty
  analyze_failure_patterns
  profile_inference
  profile_trajectory
  # ROS2 observability
  get_ros2_topic_health
  get_ros2_node_status
  get_ros2_alerts
)

# Verify feature is available
info "Checking Agent Builder availability..."
code=$(curl $CURL_FLAGS -u "$ES_USER:$ES_PASSWORD" -H "kbn-xsrf: true" \
       -o /dev/null -w '%{http_code}' \
       "$KB_URL/api/agent_builder/agents")
if [[ "$code" != "200" ]]; then
  fail "Agent Builder returned HTTP $code. License likely doesn't permit the feature. Run 01-start-trial.sh first."
fi
ok "Agent Builder is available"

# ---------------------------------------------------------------------------
# Step 1: Register each MCP tool individually.
#
# Agent Builder requires tools to exist before they can be referenced by an
# agent. Each tool is registered with the MCP connector ID and bare tool name;
# the resulting tool ID uses the "robot.<tool_name>" format.
# ---------------------------------------------------------------------------
info "Registering ${#MCP_TOOLS[@]} MCP tools with connector '$MCP_CONNECTOR_ID'..."
tool_failures=0

for tool_name in "${MCP_TOOLS[@]}"; do
  tool_id="robot.${tool_name}"
  payload=$(cat <<TOOLJSON
{"id": "${tool_id}", "type": "mcp", "configuration": {"connector_id": "${MCP_CONNECTOR_ID}", "tool_name": "${tool_name}"}}
TOOLJSON
  )

  resp=$(kb_post "api/agent_builder/tools" "$payload")

  if echo "$resp" | grep -q '"id"'; then
    ok "  $tool_id"
  elif echo "$resp" | grep -q 'already exists\|conflict\|409'; then
    ok "  $tool_id (already exists)"
  else
    warn "  $tool_id — unexpected response: $resp"
    tool_failures=$((tool_failures + 1))
  fi
done

if [[ "$tool_failures" -gt 0 ]]; then
  warn "$tool_failures tool(s) may not have registered correctly — review output above"
else
  ok "All ${#MCP_TOOLS[@]} MCP tools registered"
fi

# ---------------------------------------------------------------------------
# Step 2: Create the agent, referencing tools by their connector-prefixed IDs.
# ---------------------------------------------------------------------------

# Build the tool_ids JSON array: ["robot.execute_task", "robot.stop_robot", ...]
tool_ids_json=$(printf '"%s", ' "${MCP_TOOLS[@]/#/robot.}")
tool_ids_json="[${tool_ids_json%, }]"

cat > /tmp/so101-agent.json <<'AGENTJSON'
{
  "id": "AGENT_ID_PLACEHOLDER",
  "name": "AGENT_NAME_PLACEHOLDER",
  "description": "AI operations assistant for the SO-101 robot arm. Commands the robot, diagnoses failures across GPU kernels / CLIP embeddings / VLM evaluations / trajectory analysis, and monitors the full observability stack.",
  "configuration": {
    "instructions": "You are an AI operations assistant for a physical SO-101 robotic arm. You can command the robot, diagnose failures, and monitor the full observability stack — from GPU kernels to ROS2 topics to VLM evaluations.\n\nDIAGNOSTIC REASONING — when asked to diagnose failures or investigate problems:\n1. Start with query_inference_stats to see overall success/failure rates by model or condition.\n2. Run analyze_failure_patterns to identify the dominant failure mode and whether failures cluster by scene novelty, hardware issues, or policy quality.\n3. Run diagnose_trial to classify the failure type: planning_failure (model computes the same trajectory regardless of outcome), compute_anomaly (GPU divergence between success and failure), or hardware_failure (zero steps — camera or motor hang).\n4. Based on classification:\n   - planning_failure: Use check_scene_novelty to test whether failed scenes are outside the training distribution. Use profile_trajectory to check motion quality.\n   - compute_anomaly: Use get_gpu_memory_status and get_control_loop_status to check for thermal throttling, memory pressure, or deadline misses. Use profile_inference for timing.\n   - hardware_failure: Use get_ros2_topic_health and get_ros2_alerts to identify camera or motor issues.\n5. Synthesize findings into a root cause with evidence chain and specific recommendation.\n\nALWAYS provide the evidence chain — which tools you called, what they showed, and how the signals connect.\n\nOPERATIONS:\n- execute_task: Command the robot (triggers REAL physical movement)\n- evaluate_cycle: Score workspace with Qwen3-VL vision AI\n- set_condition / get_condition: Tag environment conditions\n- swap_model: Switch VLA model checkpoint\n- stop_robot: Halt inference\n\nAlways use tools — never just describe what you would do.",
    "tools": [
      { "tool_ids": TOOL_IDS_PLACEHOLDER }
    ]
  }
}
AGENTJSON

# Substitute placeholders with shell variables (heredoc is single-quoted to
# preserve newlines in the instructions, so we sed the placeholders).
sed -i'' \
  -e "s|AGENT_ID_PLACEHOLDER|${AGENT_ID}|" \
  -e "s|AGENT_NAME_PLACEHOLDER|${AGENT_NAME}|" \
  -e "s|TOOL_IDS_PLACEHOLDER|${tool_ids_json}|" \
  /tmp/so101-agent.json

info "Creating agent '$AGENT_NAME'..."
resp=$(kb_post "api/agent_builder/agents" "$(cat /tmp/so101-agent.json)")
echo "$resp" | python3 -m json.tool

if echo "$resp" | grep -q "\"id\""; then
  ok "Agent created"
else
  warn "Agent creation may have failed — review response above"
fi

rm -f /tmp/so101-agent.json
