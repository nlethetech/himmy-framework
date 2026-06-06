# Connect an agent to any REST API — no code

Give an agent a custom tool by **describing an HTTP call in `agent.yaml`** — no Python, no
`tools.py`. The framework builds the request (URL, path placeholders, query/body/header
args, env-backed auth) and validates the model's arguments against an auto-derived schema.

## Run it

```bash
pip install -e .
ollama pull qwen2.5:3b-instruct
cd examples/http-tool

himmy run -f agent.yaml -p "What is the USD to EUR exchange rate? Call exchange_rate with base=USD and symbols=EUR."
```
```
⚙ exchange_rate {"base": "USD", "symbols": "EUR"}
agent › The current USD to EUR exchange rate is 0.85911.
```

That number comes straight from the live API — verified on `qwen2.5:3b-instruct`.

## The `http_tools` block

```yaml
http_tools:
  - name: exchange_rate                 # the tool name the model calls
    description: ...                     # tell the model when/how to use it
    base_url: https://api.frankfurter.dev
    path: /v1/latest                     # use {placeholders} for path args
    query: [base, symbols]               # arg names sent as ?base=…&symbols=…
    # body: [...]                        # arg names sent in the JSON body
    # headers: [...]                     # arg names sent as request headers
    # method: POST                       # defaults to GET
```

The model's argument schema is **derived automatically**: path `{placeholders}` are
**required**, and `query` / `body` / `header` names are optional — so the model knows
exactly what to pass.

## Your own (authenticated) API

Keep secrets out of the file — read the base URL and key from env vars:

```yaml
http_tools:
  - name: get_order
    description: Look up an order by id.
    base_url_env_var: MYAPI_URL          # export MYAPI_URL=https://api.mycompany.com
    path: /orders/{order_id}
    auth: { type: bearer, env_var: MYAPI_KEY }
```
```bash
export MYAPI_URL=https://api.mycompany.com
export MYAPI_KEY=sk-...
himmy run -f agent.yaml -p "What's the status of order 12345?"
```

Auth `type` can be `bearer`, `header` (with `header_name`), `basic`, or `none`. The secret
is read from `env_var` at call time and never stored on the model.

## Notes

- **Redirects are not followed** (an SSRF safety guard). If an API returns a 3xx, you'll
  get a clear error telling you to point `base_url`/`path` at the final location — e.g.
  `api.frankfurter.app` redirects, so this example uses `api.frankfurter.dev`.
- Hosts are pinned and args are schema-validated, so a declarative tool can't be coerced
  into hitting a different host or smuggling extra parameters.
