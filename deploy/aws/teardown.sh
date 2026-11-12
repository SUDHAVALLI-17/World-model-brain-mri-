#!/usr/bin/env bash
# Delete everything setup_aws.sh created. Asks before doing anything.
#   ./teardown.sh             (keeps the S3 bucket with your models)
#   ./teardown.sh --all       (also empties and deletes the bucket)
set -euo pipefail
cd "$(dirname "$0")"
source ./state.env
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""
echo "This deletes: instance $INSTANCE, security group $SG, role $ROLE, secret $SECRET_NAME, log group $LOG_GROUP"
[ "${1:-}" = "--all" ] && echo "               and bucket $BUCKET with everything in it"
read -r -p "Type the project name ($NAME) to confirm: " ok
[ "$ok" = "$NAME" ] || { echo "cancelled"; exit 1; }

aws ec2 terminate-instances --instance-ids "$INSTANCE" >/dev/null || true
aws ec2 wait instance-terminated --instance-ids "$INSTANCE" || true
if [ -n "${EIP_ALLOC:-}" ]; then aws ec2 release-address --allocation-id "$EIP_ALLOC" || true; fi
aws ec2 delete-security-group --group-id "$SG" || true
aws iam remove-role-from-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" || true
aws iam delete-instance-profile --instance-profile-name "$ROLE" || true
aws iam delete-role-policy --role-name "$ROLE" --policy-name "${NAME}-app" || true
aws iam detach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore || true
aws iam delete-role --role-name "$ROLE" || true
aws secretsmanager delete-secret --secret-id "$SECRET_NAME" --recovery-window-in-days 7 >/dev/null || true
aws logs delete-log-group --log-group-name "$LOG_GROUP" || true
aws budgets delete-budget --account-id "$ACCOUNT" --budget-name "$NAME-monthly" 2>/dev/null || true
if [ "${1:-}" = "--all" ]; then aws s3 rb "s3://$BUCKET" --force || true; fi
echo "done. The secret can be restored for 7 days: aws secretsmanager restore-secret --secret-id $SECRET_NAME"
