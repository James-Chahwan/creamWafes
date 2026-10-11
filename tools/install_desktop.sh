#!/usr/bin/env bash
# Put creamWafes in your desktop's app menu and dock, with its icon (Linux, this user only).
#   tools/install_desktop.sh            install / update
#   tools/install_desktop.sh --remove   take it out again
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA=${XDG_DATA_HOME:-$HOME/.local/share}
APPS=$DATA/applications
ICONS=$DATA/icons/hicolor

if [ "${1:-}" = "--remove" ]; then
    rm -f "$APPS/creamwafes.desktop"
    for s in 128 256 512; do rm -f "$ICONS/${s}x${s}/apps/creamwafes.png"; done
    echo "Removed creamWafes from the app menu."
    exit 0
fi

UV=$(command -v uv || echo "$HOME/.local/bin/uv")
for s in 128 256 512; do
    install -Dm644 "$ROOT/app/creamwafes/assets/app-icon-$s.png" "$ICONS/${s}x${s}/apps/creamwafes.png"
done
mkdir -p "$APPS"
sed "s|@EXEC@|$UV run --project \"$ROOT/app\" creamwafes|" "$ROOT/packaging/creamwafes.desktop" > "$APPS/creamwafes.desktop"
update-desktop-database "$APPS" 2>/dev/null || true
gtk-update-icon-cache -q "$ICONS" 2>/dev/null || true
echo "creamWafes is in your app menu (search for it). Undo: tools/install_desktop.sh --remove"
