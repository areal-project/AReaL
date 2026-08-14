#!/bin/bash
# 双节点连通性探测 v2: 每节点写独立文件, worker 公布 POD_IP, master 用 IP 直连(绕过 DNS).
# 同一脚本两节点都跑, 靠 POD_NAME/RANK 分流.
set -uo pipefail

SHARE_DIR="/storage/openpsi/users/yl/agent-memory/MemRL/logs/probe_share"
mkdir -p "$SHARE_DIR" 2>/dev/null || true

# role detection
if [ -n "${RANK:-}" ]; then NODE_RANK="$RANK"
elif [[ "${POD_NAME:-}" =~ [Mm]aster ]]; then NODE_RANK=0
elif [[ "${POD_NAME:-}" =~ [Ww]orker ]]; then NODE_RANK=1
else NODE_RANK=0; fi

SELF_IP="${POD_IP:-${NODE_IP:-unknown}}"
echo "=========================================="
echo "PROBE v2 | $(date) | ROLE_RANK=$NODE_RANK POD_NAME=${POD_NAME:-unset} IP=$SELF_IP WORLD_SIZE=${WORLD_SIZE:-unset}"
echo "=========================================="

PROBE_PORT=8000

if [ "$NODE_RANK" != "0" ]; then
    # ---------------- WORKER ----------------
    echo "[WORKER] executing. IP=$SELF_IP hostname=$(hostname 2>/dev/null)"
    # publish own IP so master can find us without DNS
    echo "$SELF_IP" > "$SHARE_DIR/worker_ip.txt"
    echo "[WORKER] wrote IP to $SHARE_DIR/worker_ip.txt"
    cd /tmp; echo "worker-alive-$SELF_IP" > /tmp/probe_marker.txt
    python3 -m http.server ${PROBE_PORT} --bind 0.0.0.0 &
    SRV_PID=$!
    echo "[WORKER] http.server pid=$SRV_PID on 0.0.0.0:$PROBE_PORT"
    for i in $(seq 1 90); do   # up to 30 min
        if [ -f "$SHARE_DIR/DONE" ]; then echo "[WORKER] saw DONE, self-terminating after ~$((i*20))s"; kill $SRV_PID 2>/dev/null || true; exit 0; fi
        if ! kill -0 $SRV_PID 2>/dev/null; then echo "[WORKER] http.server died"; exit 1; fi
        sleep 20
    done
    echo "[WORKER] timed out waiting DONE"; kill $SRV_PID 2>/dev/null || true; exit 0
fi

# ---------------- MASTER ----------------
rm -f "$SHARE_DIR/DONE" "$SHARE_DIR/worker_ip.txt" 2>/dev/null || true
echo "[MASTER] executing. IP=$SELF_IP"

# also test hostname inference for reference
WORKER_HOST=$(printf '%s' "${POD_NAME:-}" | sed 's/-master-/-worker-/')
echo "[MASTER] inferred WORKER_HOST=$WORKER_HOST"
echo "[MASTER] getent hostname test:"; getent hosts "$WORKER_HOST" 2>/dev/null && echo "  -> hostname RESOLVES" || echo "  -> hostname does NOT resolve (expected, will use IP)"

# wait for worker to publish its IP
echo "[MASTER] waiting for worker to publish IP via $SHARE_DIR/worker_ip.txt ..."
WORKER_IP=""
for i in $(seq 1 90); do   # up to 30 min
    if [ -f "$SHARE_DIR/worker_ip.txt" ]; then
        WORKER_IP=$(cat "$SHARE_DIR/worker_ip.txt" 2>/dev/null | tr -d '[:space:]')
        [ -n "$WORKER_IP" ] && { echo "[MASTER] got WORKER_IP=$WORKER_IP after ~$((i*20))s"; break; }
    fi
    echo "[MASTER] waiting worker_ip... iter=$i"
    sleep 20
done

if [ -z "$WORKER_IP" ]; then
    echo "[MASTER] ===== FAIL: worker never published IP (worker likely didn't run) ====="
    touch "$SHARE_DIR/DONE"; exit 0
fi

echo "[MASTER] === connectivity test: curl http://$WORKER_IP:$PROBE_PORT/probe_marker.txt ==="
CONNECTED=0
for i in $(seq 1 60); do   # up to 20 min
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://${WORKER_IP}:${PROBE_PORT}/probe_marker.txt" 2>/dev/null || true)
    if [ "$CODE" = "200" ]; then
        BODY=$(curl -s "http://${WORKER_IP}:${PROBE_PORT}/probe_marker.txt" 2>/dev/null || true)
        echo "[MASTER] SUCCESS via IP after ~$((i*20))s. body=[$BODY]"
        CONNECTED=1; break
    fi
    echo "[MASTER] curl worker IP... iter=$i code=[$CODE]"
    sleep 20
done

if [ "$CONNECTED" = "1" ]; then
    echo "[MASTER] ===== DUAL-NODE (via IP): PASS ====="
else
    echo "[MASTER] ===== DUAL-NODE (via IP): FAIL ====="
fi
touch "$SHARE_DIR/DONE"
echo "[MASTER] done $(date). Exiting to observe job lifecycle."
exit 0
