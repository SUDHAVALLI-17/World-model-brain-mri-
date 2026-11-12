#!/bin/bash
# EC2 first-boot script (rendered by setup_aws.sh). Installs Docker, pulls the app and
# models from the private S3 bucket, and starts the containers. Log: /var/log/brainwm-setup.log
set -euxo pipefail
exec > >(tee -a /var/log/brainwm-setup.log) 2>&1
export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y ca-certificates curl unzip unattended-upgrades
curl -fsSL https://get.docker.com | sh                       # Docker Engine + compose plugin
curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
unzip -q -o /tmp/awscli.zip -d /tmp && /tmp/aws/install --update

# 4 GB swap as a safety margin for preprocessing
if [ ! -f /swapfile ]; then fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile \
  && echo '/swapfile none swap sw 0 0' >> /etc/fstab; fi

mkdir -p /opt/brainwm
cat > /opt/brainwm/fetch.sh <<'EOF'
#!/bin/bash
# Download the current app bundle and models from S3 (also used by update_app.sh)
set -euo pipefail
aws s3 cp s3://__BUCKET__/bundle/app.tar.gz /tmp/app.tar.gz --region __REGION__
rm -rf /opt/brainwm/src /opt/brainwm/deploy
tar xzf /tmp/app.tar.gz -C /opt/brainwm
aws s3 sync s3://__BUCKET__/models/ /opt/brainwm/models/ --region __REGION__ --delete
EOF

cat > /opt/brainwm/start.sh <<'EOF'
#!/bin/bash
# Write .env (re-reads the public IP: it changes after stop/start without an Elastic IP) and start
set -euo pipefail
DOMAIN="__DOMAIN__"; TLS="__TLS__"
if [ -z "$DOMAIN" ]; then
  T=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
  DOMAIN=$(curl -s -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/public-ipv4)
  TLS=internal
fi
umask 077
cat > /opt/brainwm/deploy/.env <<ENV
DOMAIN=$DOMAIN
TLS=$TLS
SECRET_ID=__SECRET_NAME__
AWS_REGION=__REGION__
LOG_GROUP=__LOG_GROUP__
UPLOAD_BUCKET=__UPLOAD_BUCKET__
ENV
cd /opt/brainwm/deploy
docker compose -f docker-compose.yml -f docker-compose.aws.yml --env-file .env up -d --build --remove-orphans
EOF
chmod 700 /opt/brainwm/fetch.sh /opt/brainwm/start.sh

cat > /etc/systemd/system/brainwm.service <<'EOF'
[Unit]
Description=Brain-MRI world model web app
After=docker.service network-online.target
Requires=docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/brainwm/start.sh
[Install]
WantedBy=multi-user.target
EOF

/opt/brainwm/fetch.sh
systemctl daemon-reload
systemctl enable --now brainwm.service
echo "brainwm setup finished"
