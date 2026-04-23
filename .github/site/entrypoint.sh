#!/bin/sh
# Ensure data directories exist on the persistent volume.
# The volume mount replaces /data at runtime, so these can't
# be created at build time.
mkdir -p /data/coverage /data/bench /data/mutants /data/dudect /data/tacet
exec nginx -g 'daemon off;'
