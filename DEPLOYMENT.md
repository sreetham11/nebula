# Deploying to Google Cloud Run

One container serves both the API and the dashboard, so you get **one URL** —
that is the link to submit as the hosted prototype.

You do not need Docker or the gcloud CLI on your laptop. **Cloud Shell** runs
in the browser with both already installed.

---

## 1. Open Cloud Shell

From the Hackathon Portal → your workspace URL → Google Cloud console
(use an Incognito window and the team credentials from the portal).

In the console, click the **terminal icon** in the top-right toolbar
(“Activate Cloud Shell”). Wait for the prompt.

## 2. Get the code and set the project

```bash
git clone https://github.com/sreetham11/nebula.git
cd nebula
git checkout feat/ai-integrations

# The project id is on the portal page with your credentials; this also
# prints it if the console already selected one.
gcloud config get-value project
# If it is empty or wrong:
# gcloud config set project YOUR_PROJECT_ID
```

## 3. Enable the two services the deploy needs

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
```

## 4. Deploy

Cloud Build builds the image from the `Dockerfile` and Cloud Run serves it.
The build trains the models, so give it a generous timeout.

```bash
gcloud run deploy railpulse \
  --source . \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --port 8080
```

- `--region asia-southeast1` is Singapore — lowest latency for the demo.
- `--allow-unauthenticated` makes it publicly reachable, which a judge
  following your submitted link needs.
- `--memory 2Gi` — torch plus four models does not fit in the 512Mi default.

Answer `Y` if it offers to create an Artifact Registry repository.

**First build takes roughly 8–15 minutes** (installing torch, generating the
dataset, training). If Cloud Build stops at the default 10-minute cap:

```bash
gcloud config set builds/timeout 1800
```

then re-run the deploy.

When it finishes it prints:

```
Service URL: https://railpulse-XXXXXXXX-as.a.run.app
```

**That URL is your prototype link.** Open it — the dashboard loads directly.

## 5. Optional: the AI features

The Fleet Assistant, Maintenance Copilot and Investigate Fault need keys.
Everything else — every risk level, the twin, validation, the calculator,
CSV scoring — works without them, and each feature degrades to a message
saying the key is not configured.

```bash
gcloud run services update railpulse \
  --region asia-southeast1 \
  --set-env-vars ANTHROPIC_API_KEY=sk-ant-...,EXA_API_KEY=...
```

> Keys set this way are visible to anyone with console access to the
> project. For a hackathon demo that is fine; for anything longer-lived use
> Secret Manager (`--set-secrets`).

## 6. Check it

```bash
URL=$(gcloud run services describe railpulse --region asia-southeast1 --format='value(status.url)')
curl -s "$URL/health"
curl -s -o /dev/null -w 'dashboard %{http_code}\n' "$URL/"
curl -s "$URL/model-validation"
```

---

## If something breaks

**Build fails on the torch line.** The CPU index may not carry that exact
version. Drop `--index-url https://download.pytorch.org/whl/cpu` from the
Dockerfile and rebuild — the image gets much bigger but installs fine.

**Container fails to start / “failed to listen on PORT”.** Read the logs:

```bash
gcloud run services logs read railpulse --region asia-southeast1 --limit 50
```

**Dashboard loads but every panel says unavailable.** The API half is not
answering. Check `/health` and then the logs above.

**Out of memory.** Raise it: `--memory 4Gi`.

## Redeploying after a change

```bash
git pull
gcloud run deploy railpulse --source . --region asia-southeast1 --allow-unauthenticated --memory 2Gi --cpu 2
```
