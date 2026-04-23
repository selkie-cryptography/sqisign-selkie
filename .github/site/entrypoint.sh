#!/bin/sh
# Ensure data directories exist on the persistent volume.
# The volume mount replaces /data at runtime, so these can't
# be created at build time.
mkdir -p /data/coverage /data/bench /data/mutants /data/dudect /data/tacet /data/kat /data/deny /data/unsafe /data/size /data/docs /data/msrv /data/panic /data/fuzz
exec nginx -g 'daemon off;'
