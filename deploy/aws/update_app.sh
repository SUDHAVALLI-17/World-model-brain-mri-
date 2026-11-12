#!/usr/bin/env bash
# Push new code and/or models to the running server, then rebuild and restart the app.
#   ./update_app.sh            (after retraining: re-run deploy/package_models.py first)
#   ./update_app.sh --secrets  (also upload a changed secrets.json, e.g. a new user)
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
source ./state.env
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""
REPO=$(cd ../.. && pwd)
[ -f "$REPO/models/manifest.json" ] || { echo "no models/manifest.json"; exit 1; }

tar -C "$REPO" --exclude='__pycache__' --exclude='deploy/secrets.json' --exclude='deploy/aws/config.env' \
  --exclude='deploy/aws/rendered' --exclude='deploy/.env' -czf /tmp/${NAME}-app.tar.gz src deploy
aws s3 cp /tmp/${NAME}-app.tar.gz "s3://$BUCKET/bundle/app.tar.gz" --only-show-errors
aws s3 sync "$REPO/models" "s3://$BUCKET/models/" --delete --only-show-errors
if [ "${1:-}" = "--secrets" ]; then
  aws secretsmanager put-secret-value --secret-id "$SECRET_NAME" --secret-string "file://$SECRETS_JSON" >/dev/null
  echo "secret updated"
fi

CMD=$(aws ssm send-command --instance-ids "$INSTANCE" --document-name AWS-RunShellScript \
  --comment "brainwm update" --parameters 'commands=["/opt/brainwm/fetch.sh","/opt/brainwm/start.sh"]' \
  --query Command.CommandId --output text)
echo "update sent ($CMD); waiting for the rebuild..."
aws ssm wait command-executed --command-id "$CMD" --instance-id "$INSTANCE" || true
aws ssm get-command-invocation --command-id "$CMD" --instance-id "$INSTANCE" \
  --query '[Status,StandardErrorContent]' --output text | tail -5
