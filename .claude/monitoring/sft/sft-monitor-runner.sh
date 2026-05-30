#!/bin/bash
# Tier-1 (Sonnet) hourly health check + Tier-2 (Opus) on-alert recovery
# for SkyPilot sft-full training on BlueVela.
#
# Pod-restart resilient: all state in $HOME (NFS), and ~/.bashrc auto-launches
# this on shell login if not already running.
#
# Manual start:    nohup ~/sft-monitor-runner.sh > ~/sky-monitor-logs/runner.log 2>&1 & disown
# Stop:            pkill -f sft-monitor-runner.sh
# Disable autostart: edit/remove the snippet in ~/.bashrc

set -u

# Tunable paths. Override either var in the environment to relocate the
# scripts/state dir or point at a different SkyPilot checkout.
: "${MONITOR_HOME:=$HOME}"
: "${SKYPILOT_DIR:=$HOME/skypilot}"
export MONITOR_HOME SKYPILOT_DIR

LOG_DIR=$MONITOR_HOME/sky-monitor-logs
CHECK_PROMPT=$MONITOR_HOME/sft-monitor-check.txt
RECOVER_PROMPT=$MONITOR_HOME/sft-monitor-recover.txt
INTERVAL=3600   # 1 hour
KEEP_CHECKS=72  # keep last ~3 days of hourly checks

# Render a prompt file by substituting ONLY ${MONITOR_HOME} and ${SKYPILOT_DIR}.
# Other $VARS (e.g. $HOME in example bash blocks inside the prompt) are left
# literal for Claude to interpret at tool-call time. Uses bash parameter
# expansion so we don't depend on `envsubst` being installed.
render_prompt() {
  local content
  content=$(< "$1")
  content=${content//'${MONITOR_HOME}'/$MONITOR_HOME}
  content=${content//'${SKYPILOT_DIR}'/$SKYPILOT_DIR}
  printf '%s' "$content"
}

mkdir -p "$LOG_DIR/launch-archive"

# Heartbeat file confirms runner is alive
echo "$(date)  runner started, pid=$$" >> "$LOG_DIR/runner-heartbeat.log"

while true; do
  # Stop the loop if training is already marked complete (e.g. from a previous
  # iteration, or carried over from before a pod restart). No Claude call needed.
  if compgen -G "$LOG_DIR/COMPLETE-*.md" >/dev/null; then
    echo "$(date)  COMPLETE marker present, runner exiting" >> "$LOG_DIR/runner-heartbeat.log"
    exit 0
  fi

  TS=$(date +%Y%m%d-%H%M%S)

  # Clear stale alert from previous iteration's recovery (if any)
  rm -f "$LOG_DIR/ALERT.txt"

  # Tier 1 — Sonnet quick check (~30-60s, cheap)
  claude -p "$(render_prompt "$CHECK_PROMPT")" \
    --model claude-sonnet-4-6 \
    --dangerously-skip-permissions \
    > "$LOG_DIR/check-${TS}.md" 2>&1

  # If this iteration's check declared COMPLETE, exit immediately rather than
  # waiting another full INTERVAL. Skip the recovery branch — it isn't needed.
  if compgen -G "$LOG_DIR/COMPLETE-*.md" >/dev/null; then
    echo "$(date)  COMPLETE marker written by health check, runner exiting" >> "$LOG_DIR/runner-heartbeat.log"
    exit 0
  fi

  # Tier 2 — Opus recovery (only when Sonnet wrote ALERT.txt)
  if [ -f "$LOG_DIR/ALERT.txt" ]; then
    echo "$(date)  ALERT detected, invoking Opus recovery" >> "$LOG_DIR/runner-heartbeat.log"
    claude -p "$(render_prompt "$RECOVER_PROMPT")" \
      --model claude-opus-4-7 \
      --dangerously-skip-permissions \
      > "$LOG_DIR/recover-${TS}.md" 2>&1
    echo "$(date)  recovery agent exited" >> "$LOG_DIR/runner-heartbeat.log"
  fi

  # Trim old check files
  ls -1t "$LOG_DIR"/check-*.md 2>/dev/null | tail -n +$((KEEP_CHECKS+1)) | xargs -r rm -f

  sleep "$INTERVAL"
done
