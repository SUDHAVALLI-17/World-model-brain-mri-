# Deploying the web app on AWS

This puts the trained model behind a login-protected HTTPS website on one EC2 server in the
Mumbai region (ap-south-1). One script creates everything. Other scripts update the app,
stop/start the server to save money, and delete everything at the end.

> **Research prototype, not for diagnosis.** The page shows this banner.
> **No ADNI images on the server.** The ADNI Data Use Agreement does not allow redistribution,
> so the server holds only the trained weights. Test it with scans you upload yourself and
> keep `STORE_UPLOADS=0` (the default), so uploads are deleted right after analysis.

## What gets built

```
 Browser ──HTTPS 443──▶ Security group (443 + 80 only, no SSH)
                         │
                         ▼
   EC2 t3.large, Ubuntu 24.04, encrypted disk, IMDSv2 only, no key pair
   ┌────────────────────────────────────────────────────────────┐
   │ Caddy (TLS 1.2/1.3, auto certificate, HSTS, 410 MB limit)   │
   │   └─▶ app container: FastAPI + model, non-root, read-only,  │
   │       no Linux capabilities, uploads only in RAM-backed /tmp │
   └────────────────────────────────────────────────────────────┘
        │ IAM role (no keys on disk)
        ├──▶ S3 bucket (private, KMS-encrypted, TLS-only): code bundle + model weights
        ├──▶ Secrets Manager: session key, HMAC key, user password hashes
        └──▶ CloudWatch Logs: JSON audit log (who uploaded what hash, result), 30-day retention
   Admin access: AWS Systems Manager Session Manager (no open SSH port)
   Cost guard: AWS Budgets e-mail alert
```

## Before you start (one time)

1. **AWS account.** Create one at aws.amazon.com. Then:
   * Turn on MFA for the root user. After that, don't use the root user again.
   * In IAM Identity Center (or IAM), create a user for yourself with
     `AdministratorAccess`. Use that user for everything below. `setup_aws.sh` refuses
     to run as root.
   * Check whether your college gets credits. Ask about AWS Academy / AWS Educate. New
     accounts also get starter credits; check the current offer on the sign-up page.
2. **Install the AWS CLI v2** and log in:
   ```bash
   aws configure sso      # or: aws configure  (access key of your IAM user)
   aws sts get-caller-identity   # must show your IAM user, not root
   ```
3. **Python environment** with the project requirements (you already have it for training):
   `pip install -r requirements.txt -r deploy/requirements-app.txt`

## Step 1: package the trained model

`run_all.sh` does this for you at the end. To do it by hand after training (see the main README):

```bash
python deploy/package_models.py --classifiers "runs/p1_full/fold*/best.pt" \
    --world_model runs/pretrain_p1/last.pt --out models --version v1 \
    --trained_on "ADNI, N subjects, 5-fold, combined adversarial"
```

`p1_full` is the combined model: it was trained against joint image + metadata attacks, with
scanner debiasing, on top of the Trajectory-TRADES world model.

This writes `models/` with a `manifest.json` of SHA-256 hashes. The app refuses to start if
any weight file has been changed.

## Step 2: make the keys and the first login

```bash
cd deploy
python -m app.security make-secrets --out secrets.json            # asks for the admin password
python -m app.security add-user --file secrets.json --user student1 --role user
```

`secrets.json` holds random keys and scrypt password hashes, never the passwords themselves.
Never commit it or share it.

## Step 3 (optional but recommended): test on your laptop

```bash
cd deploy
MODEL_DIR=../models docker compose -f docker-compose.yml -f docker-compose.local.yml up --build
# open https://localhost  (self-signed certificate: accept the warning)
```

Or run it without Docker:
`cd deploy && SECRETS_FILE=secrets.json MODEL_DIR=../models COOKIE_SECURE=0 SRC_DIR=../src uvicorn app.main:app`,
then open http://127.0.0.1:8000.

## Step 4: deploy

```bash
cd deploy/aws
cp config.env.example config.env
nano config.env        # set ALERT_EMAIL; optionally DOMAIN/TLS_EMAIL, ALLOWED_CIDR
./setup_aws.sh
```

The script prints the server address. The first boot installs Docker and builds the image, so
**allow 10–15 minutes**. Watch progress with
`aws logs tail /brainwm/app --follow --region ap-south-1`. The boot log on the server itself
is `/var/log/brainwm-setup.log`.

**HTTPS choices**

| Setting | What you get |
|---|---|
| `DOMAIN` empty (default) | `https://<server-ip>` with a self-signed certificate. Browsers warn once; the traffic is still encrypted. Good enough for a demo. |
| `DOMAIN=bwm.example.in`, `TLS_EMAIL=you@…` | A free, trusted Let's Encrypt certificate. Create a DNS **A record** `bwm.example.in → server IP` at your domain provider. Set `ALLOCATE_EIP=1` so the IP never changes. |

`ALLOWED_CIDR` limits who can open the site. For example, set it to your college's public IP
range (`curl ifconfig.me` shows yours) so only campus users can reach it.

## Step 5: check it

1. Open the site and log in. Wrong passwords are rate-limited (5 per minute per IP).
2. Upload a NIfTI file (`.nii` / `.nii.gz`) or a zipped DICOM series of your own scan. Fill in
   age, sex and scan date.
3. You get CN/MCI/AD probabilities, a Grad-CAM map, brain age, a 1–3 year forecast, and a
   table of security/plausibility checks. Download the signed report, then upload it to
   **Verify report**. Change any number in the JSON and verify again: it should fail.
4. To see the audit log: `./power.sh logs`

## Day-to-day commands

| Task | Command (in `deploy/aws`) |
|---|---|
| Push retrained model or code changes | `./update_app.sh` |
| Add a user / change passwords | edit `secrets.json` with `add-user`, then `./update_app.sh --secrets` |
| Stop the server (saves money) | `./power.sh stop` |
| Start it again | `./power.sh start` (prints the new IP if you have no Elastic IP) |
| Status / logs | `./power.sh status`, `./power.sh logs` |
| Shell on the server | `aws ssm start-session --target <instance-id> --region ap-south-1` (needs the Session Manager plugin) |
| Delete everything | `./teardown.sh` (keeps the bucket) or `./teardown.sh --all` |

After step 4 works, delete your local `secrets.json` copy, or keep it somewhere safe and
offline. The live copy is in Secrets Manager.

## Cost (ap-south-1, on-demand, approximate; check the AWS pricing calculator)

| Item | Running 24×7 | Stopped |
|---|---|---|
| EC2 t3.large | ≈ $65 / month (≈ $0.09 / hour) | $0 |
| Public IPv4 address | ≈ $3.6 / month | $0 (Elastic IP: still charged) |
| 30 GB gp3 disk | ≈ $2.5 / month | ≈ $2.5 / month |
| Secrets Manager (1 secret) | $0.40 / month | $0.40 |
| S3, KMS, CloudWatch Logs | well under $2 / month at this scale | same |

If you run it only for demos and reviews, about 40 hours a month, it costs roughly
**$8–10 a month**. Always run `./power.sh stop` afterwards. The budget alert e-mails you at
80 % of `MONTHLY_BUDGET_USD`.

No GPU is needed. Inference runs on CPU in about 20–40 s per scan, most of it preprocessing.
`t3.medium` (4 GB) is too small for the brain extraction step; keep `t3.large` or larger.

## How this maps to the networks & cybersecurity syllabus

| Layer | Threat | Control in this deployment |
|---|---|---|
| Network (L3/L4) | port scanning, SSH brute force | Security group allows only 443 (and 80 for certificates/redirect), optional source-IP restriction; **no SSH port at all**: admin access goes through SSM, which is authenticated by IAM and logged by CloudTrail |
| Transport (TLS) | eavesdropping, MITM | Caddy terminates TLS 1.2/1.3 with modern ciphers and HSTS; S3 bucket policy denies any non-TLS request; AWS API calls use TLS |
| Session / application | credential stuffing, CSRF, session theft | scrypt password hashes, constant-time login with dummy hash, login rate limit, `HttpOnly` + `Secure` + `SameSite=Strict` session cookie, custom-header CSRF check, CSP and other security headers, API docs disabled |
| Application input | malicious uploads, zip bombs, zip-slip | size cap before auth is even checked, per-user rate limit, zip entry/size limits, path-traversal and symlink rejection, only the T1 series is read |
| Data integrity | tampered model / poisoned weights | SHA-256 manifest checked at start-up; read-only model volume and container filesystem |
| Data integrity | tampered reports, forged metadata | HMAC-SHA256 signed reports with a verify endpoint; optional HMAC-signed metadata (age/sex/date bound to the scan hash); plausibility rules and the brain-age consistency check from Paper 1 |
| Confidentiality at rest | disk or bucket theft | encrypted EBS volume, SSE-KMS on S3, uploads kept only in tmpfs and deleted after each job |
| Identity & access | leaked keys, over-privileged server | no access keys on the server (instance role); IAM policy limited to one bucket prefix, one secret and one log group; IMDSv2 required (blocks SSRF credential theft) |
| Container | breakout, privilege escalation | non-root user (uid 10001), `cap_drop: ALL`, `no-new-privileges`, read-only root, memory limit |
| Monitoring & audit | undetected misuse | JSON audit line per request (user, IP, file SHA-256, result, check failures) to CloudWatch; CloudTrail records every AWS API call |
| Availability / cost | DoS, runaway bills | one job at a time per worker (semaphore), upload caps, budget alert, stop/start scripts |

## Troubleshooting

* **Site not reachable after 15 min.** Run `aws ssm start-session --target <id>`, then
  `sudo tail -50 /var/log/brainwm-setup.log` and
  `sudo docker compose -f /opt/brainwm/deploy/docker-compose.yml ps`.
* **`app` keeps restarting.** Usually the model files or the secret are missing. Run
  `sudo docker logs deploy-app-1`. The app prints which model file failed its hash check or
  which secret it could not read.
* **Let's Encrypt certificate not issued.** The DNS A record must point at the current IP
  and port 80 must be open. Caddy retries automatically.
* **SSM session not available.** The SSM agent needs a few minutes after first boot. The
  instance role must keep `AmazonSSMManagedInstanceCore`.
* **"no default VPC".** In the VPC console choose *Actions → Create default VPC*, then
  re-run `setup_aws.sh`. It is safe to re-run.

## Not covered (possible extensions)

* An Application Load Balancer with AWS WAF and ACM certificates. It is more robust, but
  adds about $20/month.
* Amazon Cognito with MFA instead of the built-in user list.
* GuardDuty threat detection and AWS Config rules.
* Several instances behind the load balancer, or ECS/Fargate.
