#!/usr/bin/env bash
# ./power.sh stop | start | status | logs
# A stopped instance costs nothing for compute (only its 30 GB disk, about $2.4/month).
set -euo pipefail
cd "$(dirname "$0")"
source ./state.env
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION" AWS_PAGER=""
case "${1:-status}" in
  stop)   aws ec2 stop-instances --instance-ids "$INSTANCE" >/dev/null
          aws ec2 wait instance-stopped --instance-ids "$INSTANCE"; echo "stopped" ;;
  start)  aws ec2 start-instances --instance-ids "$INSTANCE" >/dev/null
          aws ec2 wait instance-running --instance-ids "$INSTANCE"
          IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
          echo "running at $IP - the app is ready about 1 minute after boot (https://$IP or your domain)"
          [ -z "$EIP_ALLOC" ] && echo "note: without an Elastic IP the address changed; update your DNS record if you use a domain" ;;
  status) aws ec2 describe-instances --instance-ids "$INSTANCE" \
            --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress,InstanceType]' --output text ;;
  logs)   aws logs tail "$LOG_GROUP" --follow --since 30m ;;
  *)      echo "usage: $0 stop|start|status|logs"; exit 1 ;;
esac
