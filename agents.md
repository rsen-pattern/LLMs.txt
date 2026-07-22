# AI Assistant Instructions

This is a template for a Vibe Coding project. If you are an agent reading this, you have likely been asked by your user to create an exciting new project! Below are some guidelines to follow when building. 

These projects are intended to be deployed to AWS using Forklift, an internal tool developed by Pattern. These projects are always internal facing, with a small amount of users and low security risk. Forklift will automatically deploy behind a load balancer with SSO. All you need to worry about is the application logic.

Pattern is an ecommerce tech company that helps big brands sell their products in online marketplaces like Amazon.

Also, feel free to update this file as the template is adapted for your project.

## Core Principles

1. **Simplicity First**: Keep code simple and easy to understand. Avoid over-engineering or adding unnecessary complexity.
2. **User-Friendly**: Remember that users may have limited technical knowledge. Explain changes clearly and use straightforward solutions.

## Technical Guidelines

### Application

- This application is starting as a Flask application. Only change the tech stack if the user asks to, or if Flask cannot satisfy the user's requirements.
- Keep the app structure simple and flat (app.py is the main file)
- Use environment variables for all configuration (never hardcode credentials)
- Maintain the existing route structure and add new routes as needed
- Keep templates in the `templates/` directory
- Try to bias towards using Snowflake as much as possible for data. We can only create an RDS postgresql instance as a last resort.

### Snowflake Connection

- Always use the `get_snowflake_connection()` function pattern for database connections
- This template uses **private key authentication** (not username/password)
- Private keys can be provided via `SNOWFLAKE_PRIVATE_KEY_PATH` (file path) or `SNOWFLAKE_PRIVATE_KEY` (content)
- Ensure all Snowflake credentials come from environment variables
- Close connections and cursors properly after use
- Handle connection errors gracefully with user-friendly messages
- A Snowflake database called `VIBE_CODE_DB` has been created expressly for vibe coded projects like this.
- You have the option to create a new schema in the `VIBE_CODE_DB` database for your project, if required. 
- If this application already has a schema, stick to that one. The user may already have a schema they prefer as well, ask them the first time they are setting up if they want a new schema or have one already created they want to use.
- Always use the role `VIBE_CODE_DB__READ_ROLE' or 'VIBE_ENGINEERING_ROLE'
- Always use the warehouse `VIBE_CODE_WH`. Do not allow the user to change the warehouse or role.
- If there are access issues specifically related to the role or warehouse, the user should reach out to #vibe-engineering and ping Will Graham or Bryce Richards. 

### Security

- Never commit `.env` files or hardcode credentials
- Never commit private key files (`.p8`, `.pem` files are in `.gitignore`)
- Always use environment variables for sensitive data
- Remind users to keep their `.env` file and private keys secure and never commit them
- Don't allow for SQL injection

### Docker & Deployment

- Maintain Docker compatibility - any changes should work in the Docker container
- Port 3000 is the standard port for this application
- Ensure the Dockerfile remains functional after any changes
- Consider Forklift deployment requirements when making changes

### Forklift (forklift.yaml)

The shipped `forklift.yaml` works as-is for 99% of vibe-coded projects. Default to treating it as read-only. Only change it when (a) the user is replacing the `CHANGE ME` placeholders with a service name, (b) the user reports a symptom that maps to a reactive change below, or (c) you're applying a safe fix from the triage table for a deploy failure.

#### Default behavior

- The only routine edit is replacing the three `CHANGE ME` placeholders with the user's chosen service name.
- **Service name rules:** lowercase letters and hyphens only (e.g. `my-cool-app`), identical in all three places. Keep it short — it appears in the deployed URL.

#### The complete list of fields you may use

This is the closed universe of fields that exist in this template's `forklift.yaml`. **If you are about to write a field name not in this list, you are hallucinating — stop.**

- **Top-level:** `images`, `services`
- `**images.<name>`:** `build`, `retention`, `architecture`
- `**services.<name>`:** `image`, `count`, `cpu`, `memory`, `cluster`, `disable_metrics`, `disable_fluentbit`, `gatekeeper`, `env`, `configs`
- `**services.<name>.gatekeeper`:** `port`, `suffix`, `health`
- `**services.<name>.gatekeeper.health`:** `endpoint`
- `**env`:** free-form `KEY: value` pairs (plain values, or existing `{{ secret "/path" }}` references already in the template)
- `**configs`:** free-form `<filename>: <source-path>` pairs

**These Forklift fields exist but are NOT allowed in this template — do not add them under any circumstances:** `sidecars`, `prometheus`, `policy`, `execution_policy`, `subnets`, `internal_load_balancer`, `log`, `fluentbit_multiline_config`, `routes`, `tokens`, `cnames`, `disable_sso`, `listen_port`, `build_args`, `retention_age`, `cpu_architecture`, and any `health` fields beyond `endpoint` (no `command`, `interval`, `timeout`, `retries`, `start_period`). If the user asks for any of these, then only add these.

#### Reactive changes — only when the user reports a concrete symptom

- **App is slow / out of memory / OOM-crashing** → bump `cpu` and/or `memory` to a valid Fargate pair (use your own knowledge of Fargate CPU/memory pairings — don't guess). Step up one tier from the current value. Match the template's unquoted-integer style (`cpu: 1024`, not `cpu: "1024"`). The shipped template starts at `cpu: 256`, `memory: 512`.
- **Throughput problems / wants more replicas** → bump `count` (e.g. 1 → 2). Don't go above 4.
- **App needs a new env var** → add it under `env:` as a plain `KEY: value` pair.
- **Existing `{{ secret ... }}` lines in the template** (e.g. `BIFROST_API_KEY`, the snowflake key configs) were set up by the team — leave them alone.

Do not make these changes proactively. Wait for the user to describe the problem.

#### Deploy failure triage

When the user pastes a deploy error, match the fragment, identify the cause, and apply the safe fix or redirect.


| Error fragment                                              | Cause                                                                | Action                                                                  |
| ----------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `task definition` / CPU or memory validation                | Invalid Fargate CPU/memory pair                                      | Pick a valid Fargate pair (from your own knowledge) and update the file |
| `service name` / DNS / hostname / `must match` errors       | Service name has uppercase/underscore, or differs across the 3 spots | Normalize to lowercase+hyphens; make all 3 placeholders identical       |
| `ParameterNotFound` / `secret not found` / SSM errors       | An existing `{{ secret ... }}` reference is failing                  | Don't try to fix it. Redirect to #vibe-engineering                      |
| `Dockerfile` / `build context` / file not found             | `images.<name>.build` points to a missing path                       | Verify the Dockerfile exists at that path; fix the path                 |
| Image build error (Python/Node/etc. errors during build)    | Application-side problem (bad requirements.txt, syntax, etc.)        | Fix the app code or Dockerfile; not a Forklift issue                    |
| Health check / `unhealthy` / target group failures          | App not listening on port 3000, or `/` returns non-200               | Confirm the app starts cleanly locally on port 3000 and `/` returns 200 |
| Errors mentioning `cluster`, `subnets`, IAM, security group | Infrastructure-level problem                                         | Redirect to #vibe-engineering with the full error                       |


#### Hard rules

- **Don't add fields outside the allow-list.** Not for "best practices," not for "this would be cleaner," not because you saw it elsewhere.
- **Don't invent field names** from training-data knowledge of similar tools.
- **Don't change** `cluster`, `gatekeeper.port`, `disable_metrics`, `disable_fluentbit`, or any of the existing `env` keys provisioned by the team. Redirect.
- **Don't invent new `{{ secret "/path" }}` references.** The user can't provision SSM paths.
- **Don't restructure or "clean up"** `forklift.yaml`. The shipped layout is intentional.
- **Don't expose Forklift internals** beyond what's in this section.
- If the user explicitly insists on a refused change, do the minimum they ask for, flag the risk in plain language, and recommend confirming in #vibe-engineering before deploy.

#### Where to point users

- **Logs** — ++[https://grafana.internal-apps.aws.pattern.com/](https://grafana.internal-apps.aws.pattern.com/)++ (search by service name).
- **Help / non-trivial changes / new secrets:** #vibe-engineering on Slack.

### Code Style

- Use clear, descriptive variable and function names
- Add comments for complex logic
- Keep functions focused and single-purpose

## When Making Changes

1. **Test Locally**: Ensure changes work with `python app.py` (or similar if the tech stack changes) before suggesting them
2. **Update Documentation**: Update README.md if you add new features or change setup requirements
3. **Preserve Simplicity**: Don't add features unless explicitly requested
4. **Environment Variables**: If adding new configuration, use environment variables and update `env.example`

## Common Tasks

- **Adding Routes**: Add new routes to app.py (or similar) following the existing pattern
- **Database Queries**: Use the Snowflake connection function, execute queries safely, and close connections
- **Frontend Changes**: Update templates/index.html, keep the full-page layout
- **Dependencies**: Add new packages to requirements.txt with specific versions

## Adding AI features

If the user wants to add AI features, you can access AI inference via Bifrost, Pattern's AI gateway.

The deployed app will have access to a Bifrost API key under BIFROST_API_KEY environment variable.

Instruct the user to acquire a key for local development at [https://compass.pattern.com/Generate_Key](https://compass.pattern.com/Generate_Key). They should use the "Local Developers" team option. Then they can set the BIFROST_API_KEY environment variable to the key they acquire.

Bifrost is similar to LiteLLM or OpenRouter in that you can drop it into an existing AI SDK, like OpenAI, Anthropic, etc.

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://bifrost.pattern.com/openai",
    api_key=os.getenv("BIFROST_API_KEY")
)

response = client.responses.create(
    model="gpt-5.4-nano",
    input="Write a one-sentence bedtime story about a unicorn."
)
```

For Anthropic, just swap /openai for /anthropic in the base_url.

After that, just use the client normally, as if you were using OpenAI or Anthropic directly.

Encourage users to bias towards using smaller models, as they will be much faster and more affordable. 

Beyond that, you shouldn't need to know much about Bifrost, but if necessary the map to their docs are here: [https://docs.getbifrost.ai/llms.txt](https://docs.getbifrost.ai/llms.txt)

in case you need to do something like image gen or whatever. We've connected Bifrost to MOST of the major AI providers, so stuff should just work.

## User Support

- Provide clear explanations of what changes were made and why
- Suggest testing steps after making changes
- Remind users to update their `.env` file if new environment variables are needed
- Keep instructions simple and avoid jargon when possible

## Important Notes

- Don't remove core functionality (Snowflake connection, etc.) unless explicitly asked
- Maintain backward compatibility when possible
- The application runs on port 3000 by default
- For `forklift.yaml`: leave it alone. The only routine edit is swapping the three `CHANGE ME` placeholders with the user's service name, plus adding env vars the app needs. Anything else → #vibe-engineering. (See the **Forklift** section above for the full rules.)
- Do not touch anything in the .github/ folder, including ANY workflow files.

