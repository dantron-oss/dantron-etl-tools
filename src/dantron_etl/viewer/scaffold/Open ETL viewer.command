#!/bin/sh
# Double-click this file to open the ETL viewer for this project in your browser.
# Keep the window open while you use the viewer. Close it, or press ctrl-c, to stop the viewer.
cd "$(dirname "$0")" || exit 1
if command -v @@command@@ >/dev/null 2>&1; then
    @@command@@ view
else
@@fallback@@
fi
printf 'Press return to close this window. '
read -r _
