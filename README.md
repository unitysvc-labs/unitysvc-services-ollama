# UnitySVC Ollama Services

This repository publishes the Ollama service catalog for UnitySVC. Services are
generated from shared templates and per-model parameter files instead of checked
in expanded service folders.

The catalog exposes Ollama models through two channels:

- `byoe`: customer-provided Ollama-compatible endpoint. Requests are routed to
  the endpoint configured by the customer for that enrollment.
- `ollama-cloud`: Ollama's hosted API for models that are available from Ollama
  Cloud. This channel requires a real `OLLAMA_API_KEY` seller secret and is not
  suitable for shared mock credentials.

There are no separate BYOK/BYOE service names. A model is represented once, and
its access modes are channels on that service.

## Repository Layout

```text
templates/
  config.json
  provider.json
  offering.json.j2
  listing.json.j2
  code-example-ollama.py.j2
  description-byoe.md
  description-cloud.md
specs/
  ollama/
    ${model}.json
    ${model}.service.json
scripts/
  update_params.py
```

The `templates/` directory contains the shared service fragments used for every
Ollama service: provider metadata, offering/listing templates, code examples,
and channel-specific descriptions. The `specs/ollama/*.json` files are generated
params that fill those templates for each model. The
`specs/ollama/*.service.json` sidecars store backend `service_id` values for
uploaded services and should not be churned when regenerating params.

The authoritative source for the template/params file format is
`unitysvc-sellers`. This repository follows that format; it does not define the
format independently. To inspect the fully expanded service spec produced from a
template and param file, use:

```bash
usvc_seller specs expand SERVICE_NAME
```

For example:

```bash
usvc_seller specs expand ollama/llama3.3
```

## Development

Use the UnitySVC development environment:

```bash
source ~/unitysvc/.venv/bin/activate
```

Regenerate params from Ollama's model listings:

```bash
python scripts/update_params.py
```

Populate expanded specs from templates and params:

```bash
usvc_seller specs populate
```

Validate and format-check the generated specs:

```bash
usvc_seller specs format --check
usvc_seller specs validate
```

## Upload And Submit

Uploading or submitting services requires seller credentials for the target
backend. Configure the seller API key and URL for the environment you are
targeting:

```bash
export UNITYSVC_SELLER_API_KEY="svcpass_..."
export UNITYSVC_SELLER_API_URL="https://seller.staging.unitysvc.com/v1"
```

For production, use the production seller key and either omit
`UNITYSVC_SELLER_API_URL` to use the `usvc_seller` production default, or set it
explicitly:

```bash
export UNITYSVC_SELLER_API_KEY="svcpass_..."
export UNITYSVC_SELLER_API_URL="https://seller.unitysvc.com/v1"
```

Before uploading cloud-capable services, upload the seller secrets required by
the specs. This repo includes `seller.secrets.txt`, which lists the secret names
needed by the services. It intentionally does not contain a real Ollama Cloud API
key. Fill `OLLAMA_API_KEY` locally, or configure it in the GitHub repository
environment used by the shared upload workflows, then seed the seller secret
store:

```bash
usvc_seller secrets upload seller.secrets.txt
```

Upload the specs to the configured backend:

```bash
usvc_seller specs upload
```

After upload, submit services for review/activation:

```bash
usvc_seller services submit -l
```

## Notes

`scripts/update_params.py` fetches model metadata from Ollama search and the
Ollama Cloud model endpoint. Cloud model IDs are used for the `ollama-cloud`
channel routing key, while the service name remains under the `ollama/` provider
namespace.
