# GitHub App for the runner orchestrator

The orchestrator authenticates as a GitHub App so it can mint
just-in-time (JIT) runner registration tokens. PAT alternative is
not viable — JIT tokens are App-only.

The App's intended configuration lives in
[`manifest.json`](manifest.json) so the identity is declarative
rather than UI-only state.

## Create — automated via manifest (preferred)

1. Open [`create-form.html`](create-form.html) in a browser while
   signed in as a `selkie-cryptography` org owner.
   Click **Create App**.
2. GitHub renders the App creation page prefilled from `manifest.json`.
   Confirm to create.
3. After confirmation GitHub redirects to `redirect_url` with a
   `?code=<code>` query param. Exchange it once:
   ```sh
   curl -X POST -H "Accept: application/vnd.github+json" \
     https://api.github.com/app-manifests/<code>/conversions \
     | tee app-credentials.json
   ```
   Response includes `id` (App ID), `pem` (private key),
   `webhook_secret`, `html_url`, and more. **Save it once — the
   private key cannot be re-fetched.**

## Create — manual UI (fallback)

If the manifest flow doesn't work, recreate by hand:

Org settings → Developer settings → GitHub Apps → New GitHub App.

| Field | Value |
|---|---|
| Name | `sqisign-runners` (any unique) |
| Webhook URL | `https://sqisign-infra-orchestrator.fly.dev/webhook` |
| Webhook secret | `openssl rand -hex 32` (save it) |
| Repository permissions | Actions: R/W, Administration: R/W, Metadata: R |
| Organization permissions | Self-hosted runners: R/W |
| Subscribed events | Workflow job |
| Install on | Only this account |

After save: **Generate a private key**, download the `.pem` (one
chance — regen requires revoke). Note the **App ID** at the top of
the page.

## Install

App settings → Install App → `selkie-cryptography` → only
`sqisign-selkie`. Note the **installation ID** from the URL
(`/installations/<id>`).

## Push secrets to Fly

```sh
fly secrets set --app sqisign-infra-orchestrator \
  GITHUB_APP_ID=<app id> \
  GITHUB_APP_INSTALLATION_ID=<installation id> \
  GITHUB_APP_PRIVATE_KEY="$(cat *.private-key.pem)" \
  GITHUB_WEBHOOK_SECRET=<hex from above> \
  FLY_API_TOKEN="$(fly tokens create deploy --app sqisign-infra-runners --expiry 8760h)"
```

## Verify

After orchestrator is deployed, App settings → Advanced →
Recent Deliveries should show webhook deliveries returning 200.
`fly logs -a sqisign-infra-orchestrator` shows the spawn flow.
