#!/usr/bin/env bash
# Build the DFL TensorRT plugin shared library.  # RUN ON DEVICE
set -euo pipefail
PLUGIN_DIR="src/qint/plugin/dfl_plugin"

cmake -S "$PLUGIN_DIR" -B "$PLUGIN_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$PLUGIN_DIR/build" -j

SO="$PLUGIN_DIR/build/libdfl_plugin.so"
echo "Built $SO"
echo "Validate it against the reference:"
echo "  python3 scripts/validate_plugin.py --plugin $SO"
