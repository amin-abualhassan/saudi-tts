#!/usr/bin/env bash
# ops/watchdog.sh — keeps the Saudi TTS agent thinking, never stalled.
# Run it inside the tmux session "watchdog":   bash ~/saudi-tts/ops/watchdog.sh
# It never touches training jobs; it only (re)starts and nudges the Claude Code
# session that lives in the tmux session "agent".
set -u
cd "$HOME/saudi-tts" || exit 1
mkdir -p ops

LOG=ops/watchdog.log
MODE_FILE=ops/AUTH_MODE                          # "subscription" or "api"
FALLBACK="$HOME/.secrets/fallback.env"           # optional: export ANTHROPIC_API_KEY=sk-ant-api... (rung 4 only)
CHECK_SEC=300                                    # look every 5 minutes
STALE_SEC=5400                                   # nudge if PROGRESS.md is quiet for 90 minutes
NUDGE_GAP_SEC=5400                               # at most one nudge per 90 minutes
API_RETRY_SEC=43200                              # in API mode, try the subscription again after 12 h

NUDGE="Watchdog: resume work. Read PLAN.md and the last 40 lines of PROGRESS.md, re-set the /goal from docs/kickoff.md if it is not active, then continue the first unfinished stage. Append a heartbeat line to PROGRESS.md at least every 60 minutes."

[ -f "$MODE_FILE" ] || echo subscription > "$MODE_FILE"
last_nudge=0
api_since=0

log()   { echo "$(date -Is) $*" | tee -a "$LOG"; }
mode()  { cat "$MODE_FILE"; }
pane()  { tmux capture-pane -p -t agent -S -80 2>/dev/null; }
# Running = the agent pane's foreground process is not a bare shell (robust to how the claude binary names itself).
agent_running() {
  tmux has-session -t agent 2>/dev/null || return 1
  case "$(tmux display -p -t agent '#{pane_current_command}')" in
    bash|sh|zsh|dash|"") return 1 ;;
    *) return 0 ;;
  esac
}

start_agent() {
  tmux has-session -t agent 2>/dev/null || tmux new -d -s agent -c "$HOME/saudi-tts"
  local auth='source ~/.secrets/claude.env'
  if [ "$(mode)" = api ]; then
    auth='source ~/.secrets/claude.env; source ~/.secrets/fallback.env'
  fi
  log "starting agent session (auth: $(mode))"
  tmux send-keys -t agent "bash -c '$auth; export CLAUDE_CODE_RETRY_WATCHDOG=1; exec claude --continue --permission-mode auto'" Enter
  sleep 90
  tmux send-keys -t agent "$NUDGE" Enter
  last_nudge=$(date +%s)
}

log "watchdog up (mode: $(mode))"
while [ ! -f DONE ]; do
  now=$(date +%s)

  if ! agent_running; then
    start_agent
  else
    screen="$(pane)"

    # Rung 4: weekly limit on the subscription -> switch to the API key, if one is configured.
    if echo "$screen" | grep -qiE "hit your weekly limit"; then
      if [ "$(mode)" = subscription ] && [ -f "$FALLBACK" ]; then
        log "weekly limit seen -> switching agent to API key"
        echo api > "$MODE_FILE"; api_since=$now
        tmux send-keys -t agent "/exit" Enter; sleep 15
        continue
      elif [ ! -f "$FALLBACK" ]; then
        log "weekly limit seen and no fallback key configured; training continues, agent waits"
      fi
    fi

    # Model-specific weekly cap -> move to the other model and keep going.
    if echo "$screen" | grep -qiE "hit your Opus limit"; then
      log "Opus limit seen -> switching to Sonnet"
      tmux send-keys -t agent "/model sonnet" Enter; sleep 5
      tmux send-keys -t agent "$NUDGE" Enter; last_nudge=$now
    fi

    # Back to the subscription after 12 h on the API key.
    if [ "$(mode)" = api ] && [ $((now - api_since)) -gt $API_RETRY_SEC ]; then
      log "12 h on API key -> trying the subscription again"
      echo subscription > "$MODE_FILE"
      tmux send-keys -t agent "/exit" Enter; sleep 15
      continue
    fi

    # Alive but quiet (goal cleared, stuck, waiting): nudge, unless it is inside a short limit wait.
    if [ -f PROGRESS.md ]; then
      age=$((now - $(stat -c %Y PROGRESS.md)))
      if [ $age -gt $STALE_SEC ] && [ $((now - last_nudge)) -gt $NUDGE_GAP_SEC ] \
         && ! echo "$screen" | grep -qiE "limit.*reset|resets [0-9]"; then
        log "PROGRESS.md quiet for $((age / 60)) min -> nudging"
        tmux send-keys -t agent "$NUDGE" Enter
        last_nudge=$now
      fi
    fi
  fi

  sleep "$CHECK_SEC"
done
log "DONE file found; watchdog exiting"
