# Agent Builder and First Trial

This guide registers the MCP connector in Kibana, creates the Agent Builder
agent, registers all MCP tools, and runs your first trial. This is where
you go from "everything is deployed" to "the robot does what I ask."

## MCP connector

The MCP server runs as a FastMCP HTTP server on port 8888. Kibana Agent
Builder talks to it through a connector.

### Create the connector

1. In Kibana, go to **Stack Management > Connectors**
2. Click **Create connector**
3. Select **MCP** as the connector type
4. Set the URL to `http://<gpu-server>:8888/mcp`
5. Save the connector and note the UUID — you'll need it for tool
   registration

Or use the rebuild script, which does this automatically:

```bash
cd deploy/scripts/rebuild
./04-create-agent-builder-agent.sh
```

### Verify the connector

Test that the connector can reach the MCP server:

```bash
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31561/api/actions/connector/<connector-uuid>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{"params":{"subAction":"listTools","subActionParams":{}}}'
```

This should return a JSON array of all available MCP tools. If it times out
or returns an error, check that the MCP server pod is running and port 8888
is accessible.

**Important:** The subAction is `callTool`, not `toolsCall` or `run`. The
listing action is `listTools`. The testing action is `test`. These are the
only three valid subActions.

## Register MCP tools

Each MCP tool must be individually registered in Agent Builder. The rebuild
script handles all of them:

```bash
./04-create-agent-builder-agent.sh
```

To register a tool manually:

```bash
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31561/api/agent_builder/tools" \
  -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "id": "robot.<tool_name>",
    "type": "mcp",
    "configuration": {
      "connector_id": "<connector-uuid>",
      "tool_name": "<tool_name>"
    }
  }'
```

The tool ID must use the `robot.` prefix. The `tool_name` must match
exactly what the MCP server exposes.

The full list of available tools is in
[reference/mcp-tools.md](../reference/mcp-tools.md).

## Create the Agent Builder agent

The rebuild script creates the agent automatically. If you're doing it
manually, go to **Agent Builder** in Kibana, create a new agent, and add
the registered MCP tools to it.

Give the agent a system prompt that explains what the robot can do and how
to use the diagnostic tools. Something like:

> You are an AI assistant controlling a SO-101 robot arm. You have access to
> MCP tools for executing tasks, diagnosing failures, profiling GPU
> performance, and monitoring ROS 2 health. When a trial fails, use
> diagnose_trial to understand why before running another attempt.

## Your first trial

Open Kibana Agent Builder, select your agent, and type:

> Pick up the cube

The agent will:

1. Call `execute_task` with the task instruction
2. The MCP server publishes the instruction to the inference node via ROS 2
3. The VLA model runs inference for ~600 steps at 30 FPS (~20 seconds)
4. Motor commands flow to the edge node, driving the arm
5. After execution, the MCP server evaluates the result via Ollama
6. A trial document is indexed to `robot-inference`
7. The agent returns a summary with success/failure, confidence, and timing

You can also call tools directly via the connector API without Agent
Builder:

```bash
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31561/api/actions/connector/<connector-uuid>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "params": {
      "subAction": "callTool",
      "subActionParams": {
        "name": "execute_task",
        "arguments": {"task": "pick up the cube"}
      }
    }
  }'
```

### What to check after your first trial

1. **Kibana dashboards** — the Robot Operations Overview should show your
   trial result
2. **APM UI** — you should see an `execute_task` transaction trace with
   timing breakdown
3. **Elasticsearch** — query `robot-inference` for your trial document:

```bash
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31920/robot-inference/_search?size=1&sort=@timestamp:desc" \
  | python3 -m json.tool
```

### If the trial fails

Don't worry — the diagnostic system is built for this. Ask the agent:

> Diagnose the last trial

Or call the tool directly:

```bash
# diagnose_trial analyzes the most recent trial
curl -sk -u elastic:<password> \
  "https://<gpu-server>:31561/api/actions/connector/<connector-uuid>/_execute" \
  -X POST -H 'kbn-xsrf: true' -H 'Content-Type: application/json' \
  -d '{
    "params": {
      "subAction": "callTool",
      "subActionParams": {
        "name": "diagnose_trial",
        "arguments": {}
      }
    }
  }'
```

The diagnostics system fuses GPU kernel traces, CLIP distance, trajectory
analysis, and VLM evaluation to classify the failure and suggest what to
investigate. See [operate/03-diagnostics.md](../operate/03-diagnostics.md)
for the full diagnostic workflow.

## What you should have at the end

- [ ] MCP connector created in Kibana with verified connectivity
- [ ] All MCP tools registered in Agent Builder
- [ ] Agent created and configured
- [ ] First trial executed (success or failure — both are fine)
- [ ] Trial data visible in Kibana dashboards
- [ ] APM traces visible for the tool call

The deploy stage is complete. Your robot is running, observable, and
controllable through Kibana Agent Builder.

Next: [operate/01-trials.md](../operate/01-trials.md) — running trials,
interpreting results, and operating the system day to day.
