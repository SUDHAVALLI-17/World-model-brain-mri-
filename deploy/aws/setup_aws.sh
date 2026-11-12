#!/usr/bin/env bash
# Create the whole AWS deployment. Safe to re-run: existing pieces are reused.
#   cd deploy/aws && cp config.env.example config.env && (edit) && ./setup_aws.sh
# Needs: AWS CLI v2 logged in as an IAM user or SSO role (never the root user), models/ and secrets.json.
set -euo pipefail
cd "$(dirname "$0")"
die() { echo "ERROR: $*" >&2; exit 1; }
[ -f config.env ] || die "copy config.env.example to config.env and edit it"
# shellcheck disable=SC1091
source ./config.env
: "${NAME:?}" "${REGION:?}" "${INSTANCE_TYPE:?}"
[[ "$NAME" =~ ^[a-z0-9-]{3,30}$ ]] || die "NAME must be 3-30 lower-case letters, digits or -"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""
REPO=$(cd ../.. && pwd)
mkdir -p rendered
command -v aws >/dev/null || die "install AWS CLI v2 first"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
CALLER=$(aws sts get-caller-identity --query Arn --output text)
[[ "$CALLER" == *":root" ]] && die "you are logged in as the root user - create an IAM user/SSO role instead"
BUCKET="${NAME}-${ACCOUNT}-${REGION}"
SECRET_NAME="${NAME}/app"
LOG_GROUP="/${NAME}/app"
ROLE="${NAME}-ec2-role"
SG_NAME="${NAME}-https"
[ -f "$REPO/models/manifest.json" ] || die "no models/manifest.json - run deploy/package_models.py first"
[ -f "$SECRETS_JSON" ] || die "no $SECRETS_JSON - run: python -m app.security make-secrets --out deploy/secrets.json"
if [ -n "${DOMAIN:-}" ] && [ -z "${TLS_EMAIL:-}" ]; then die "set TLS_EMAIL when DOMAIN is set"; fi
echo "== account $ACCOUNT, region $REGION, bucket $BUCKET"

# ---------------------------------------------------------------- 1. private, encrypted S3 bucket
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms"},"BucketKeyEnabled":true}]}'
cat > rendered/bucket-policy.json <<EOF
{"Version":"2012-10-17","Statement":[{"Sid":"DenyInsecureTransport","Effect":"Deny","Principal":"*",
 "Action":"s3:*","Resource":["arn:aws:s3:::$BUCKET","arn:aws:s3:::$BUCKET/*"],
 "Condition":{"Bool":{"aws:SecureTransport":"false"}}}]}
EOF
aws s3api put-bucket-policy --bucket "$BUCKET" --policy file://rendered/bucket-policy.json
echo "== bucket ready (private, KMS-encrypted, TLS-only)"

# ---------------------------------------------------------------- 2. app bundle + models
tar -C "$REPO" --exclude='__pycache__' --exclude='deploy/secrets.json' --exclude='deploy/aws/config.env' \
  --exclude='deploy/aws/rendered' --exclude='deploy/.env' -czf /tmp/${NAME}-app.tar.gz src deploy
aws s3 cp /tmp/${NAME}-app.tar.gz "s3://$BUCKET/bundle/app.tar.gz" --only-show-errors
aws s3 sync "$REPO/models" "s3://$BUCKET/models/" --delete --only-show-errors
echo "== app bundle and models uploaded"

# ---------------------------------------------------------------- 3. secret
if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
  aws secretsmanager put-secret-value --secret-id "$SECRET_NAME" --secret-string "file://$SECRETS_JSON" >/dev/null
else
  aws secretsmanager create-secret --name "$SECRET_NAME" --description "brainwm session/HMAC keys and users" \
    --secret-string "file://$SECRETS_JSON" >/dev/null
fi
SECRET_ARN=$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --query ARN --output text)
echo "== secret stored: $SECRET_NAME (delete the local $SECRETS_JSON once the app works)"

# ---------------------------------------------------------------- 4. log group
aws logs create-log-group --log-group-name "$LOG_GROUP" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 30

# ---------------------------------------------------------------- 5. IAM role (least privilege)
cat > rendered/trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
UPLOAD_STMT=""
UPLOAD_BUCKET=""
if [ "${STORE_UPLOADS:-0}" = 1 ]; then
  UPLOAD_STMT=',{"Sid":"ArchiveUploads","Effect":"Allow","Action":"s3:PutObject","Resource":"arn:aws:s3:::'"$BUCKET"'/uploads/*"}'
  UPLOAD_BUCKET="$BUCKET"
fi
cat > rendered/role-policy.json <<EOF
{"Version":"2012-10-17","Statement":[
 {"Sid":"ReadBundleAndModels","Effect":"Allow","Action":"s3:GetObject",
  "Resource":["arn:aws:s3:::$BUCKET/bundle/*","arn:aws:s3:::$BUCKET/models/*"]},
 {"Sid":"ListModels","Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::$BUCKET",
  "Condition":{"StringLike":{"s3:prefix":["models/*","bundle/*"]}}},
 {"Sid":"ReadAppSecret","Effect":"Allow","Action":"secretsmanager:GetSecretValue","Resource":"$SECRET_ARN"},
 {"Sid":"WriteAppLogs","Effect":"Allow","Action":["logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"],
  "Resource":"arn:aws:logs:$REGION:$ACCOUNT:log-group:$LOG_GROUP:*"},
 {"Sid":"KmsViaS3Only","Effect":"Allow","Action":["kms:Decrypt","kms:GenerateDataKey"],"Resource":"*",
  "Condition":{"StringEquals":{"kms:ViaService":"s3.$REGION.amazonaws.com"}}}$UPLOAD_STMT
]}
EOF
aws iam get-role --role-name "$ROLE" >/dev/null 2>&1 || \
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document file://rendered/trust.json >/dev/null
aws iam put-role-policy --role-name "$ROLE" --policy-name "${NAME}-app" --policy-document file://rendered/role-policy.json
aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
if ! aws iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  echo "   waiting 20 s for IAM to propagate"; sleep 20
fi
echo "== IAM role $ROLE (S3 read, one secret, own log group, Session Manager)"

# ---------------------------------------------------------------- 6. security group (firewall)
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
[ "$VPC" != "None" ] || die "no default VPC in $REGION - create one in the VPC console (Actions > Create default VPC)"
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC" \
  --query 'SecurityGroups[0].GroupId' --output text)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "brainwm: HTTPS only, no SSH" \
    --vpc-id "$VPC" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 443 --cidr "${ALLOWED_CIDR:-0.0.0.0/0}" >/dev/null
  # port 80 only serves the Let's Encrypt challenge and a redirect to HTTPS
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null
fi
echo "== security group $SG: inbound 443 from ${ALLOWED_CIDR:-0.0.0.0/0}, 80 for certificates, no SSH"

# ---------------------------------------------------------------- 7. EC2 instance
AMI=$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)
TLS_VALUE="${TLS_EMAIL:-internal}"
sed -e "s|__BUCKET__|$BUCKET|g" -e "s|__REGION__|$REGION|g" -e "s|__SECRET_NAME__|$SECRET_NAME|g" \
    -e "s|__LOG_GROUP__|$LOG_GROUP|g" -e "s|__DOMAIN__|${DOMAIN:-}|g" -e "s|__TLS__|$TLS_VALUE|g" \
    -e "s|__UPLOAD_BUCKET__|$UPLOAD_BUCKET|g" user-data.sh.tpl > rendered/user-data.sh
INSTANCE=$(aws ec2 describe-instances --filters Name=tag:Name,Values="$NAME" \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[0].Instances[0].InstanceId' --output text)
if [ "$INSTANCE" = "None" ]; then
  INSTANCE=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile Name="$ROLE" --security-group-ids "$SG" --user-data file://rendered/user-data.sh \
    --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2,HttpEndpoint=enabled \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=$NAME}]" \
                         "ResourceType=volume,Tags=[{Key=Project,Value=$NAME}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "== launched $INSTANCE ($INSTANCE_TYPE, Ubuntu 24.04, encrypted disk, IMDSv2 only, no key pair)"
else
  echo "== instance $INSTANCE already exists - use update_app.sh to push new code/models"
fi
aws ec2 wait instance-running --instance-ids "$INSTANCE"

EIP_ALLOC=""
if [ "${ALLOCATE_EIP:-0}" = 1 ]; then
  EIP_ALLOC=$(aws ec2 describe-addresses --filters Name=tag:Project,Values="$NAME" --query 'Addresses[0].AllocationId' --output text)
  if [ "$EIP_ALLOC" = "None" ]; then
    EIP_ALLOC=$(aws ec2 allocate-address --domain vpc --tag-specifications \
      "ResourceType=elastic-ip,Tags=[{Key=Project,Value=$NAME}]" --query AllocationId --output text)
  fi
  aws ec2 associate-address --instance-id "$INSTANCE" --allocation-id "$EIP_ALLOC" >/dev/null
fi
IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# ---------------------------------------------------------------- 8. budget alert
if [ -n "${ALERT_EMAIL:-}" ] && ! aws budgets describe-budget --account-id "$ACCOUNT" --budget-name "$NAME-monthly" >/dev/null 2>&1; then
  aws budgets create-budget --account-id "$ACCOUNT" \
    --budget "{\"BudgetName\":\"$NAME-monthly\",\"BudgetLimit\":{\"Amount\":\"${MONTHLY_BUDGET_USD:-20}\",\"Unit\":\"USD\"},\"TimeUnit\":\"MONTHLY\",\"BudgetType\":\"COST\"}" \
    --notifications-with-subscribers "[{\"Notification\":{\"NotificationType\":\"ACTUAL\",\"ComparisonOperator\":\"GREATER_THAN\",\"Threshold\":80,\"ThresholdType\":\"PERCENTAGE\"},\"Subscribers\":[{\"SubscriptionType\":\"EMAIL\",\"Address\":\"$ALERT_EMAIL\"}]}]" \
    || echo "   (budget not created - set it up in Billing > Budgets)"
fi

cat > state.env <<EOF
NAME=$NAME
REGION=$REGION
ACCOUNT=$ACCOUNT
BUCKET=$BUCKET
SECRET_NAME=$SECRET_NAME
LOG_GROUP=$LOG_GROUP
ROLE=$ROLE
SG=$SG
INSTANCE=$INSTANCE
EIP_ALLOC=$EIP_ALLOC
EOF

echo
echo "================================================================"
echo " Server $INSTANCE is booting at $IP"
echo " First start installs Docker and builds the image: allow 10-15 minutes."
if [ -n "${DOMAIN:-}" ]; then
  echo " 1. Create a DNS A record:  $DOMAIN -> $IP  (then Let's Encrypt issues the certificate)"
  echo " 2. Open https://$DOMAIN"
else
  echo " Open https://$IP  (self-signed certificate: accept the browser warning)"
fi
echo " Follow progress:  aws logs tail $LOG_GROUP --follow --region $REGION"
echo " Server console:   aws ssm start-session --target $INSTANCE --region $REGION"
echo " Save money:       ./power.sh stop   (and ./power.sh start)"
echo "================================================================"
