#!/usr/bin/env bash
# Run every LayerDAW end-to-end test. The real-audio test plays into a private null sink,
# so nothing comes out of the speakers.
set -u
cd "$(dirname "$0")/../app"

SINK=layerdaw_test
MOD=""
if ! pactl list short sinks | grep -q "$SINK"; then
    MOD=$(pactl load-module module-null-sink sink_name=$SINK sink_properties=device.description=LayerDAW-test)
fi
trap '[ -n "$MOD" ] && pactl unload-module "$MOD"' EXIT

status=0
for t in test_timeline test_pads test_chords test_creative test_sounds test_song test_layer test_real_audio; do
    echo "=== $t"
    LAYERDAW_NO_ENGINE=1 PIPEWIRE_NODE=$SINK timeout 240 uv run python "tests/$t.py" 2>&1 | grep -v -E "^qt\.|propagateSizeHints|^ALSA lib"
    code=${PIPESTATUS[0]}
    if [ "$code" -eq 124 ]; then echo "!!! $t TIMED OUT (a dialog or loop is waiting) — FAILED"; status=1
    elif [ "$code" -ne 0 ]; then echo "!!! $t FAILED (exit $code)"; status=1; fi
done
exit $status
