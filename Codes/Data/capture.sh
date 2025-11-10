#!/usr/bin/env bash
set -euo pipefail

# Defaults (match values from your original snippet)
DEFAULT_FREQ=2450000000
DEFAULT_SR=20000000
DEFAULT_ANT=1
DEFAULT_LNA=24
DEFAULT_GAIN=16
DEFAULT_OUTDIR="$PWD"

usage() {
  cat <<EOF
Usage: $0 -a ANT -t DURATION_SECONDS [options]

Required:
  -a ANT               Antenna ID (integer) - will be embedded in filename
  -t DURATION          Duration in seconds (integer). Script will run hackrf_transfer for this long.

Optional:
  -f FREQ              Center frequency in Hz (default: $DEFAULT_FREQ)
  -s SR                Sample rate in samples/sec (default: $DEFAULT_SR)
  -l LNA               LNA (if used) (default: $DEFAULT_LNA)
  -g GAIN              VGA gain (default: $DEFAULT_GAIN)
  -o OUTDIR            Output directory (default: current dir)
  -n PREFIX            Filename prefix (default: capture)
  -h                   Show this help

Example:
  $0 -a 2 -t 60 -f 2450000000 -s 20000000 -o /data/captures
EOF
}

# parse args
ANT=""
DURATION=""
PREFIX="capture"
while getopts ":a:t:f:s:l:g:o:n:h" opt; do
  case ${opt} in
    a) ANT=$OPTARG ;;
    t) DURATION=$OPTARG ;;
    f) FREQ=$OPTARG ;;
    s) SR=$OPTARG ;;
    l) LNA=$OPTARG ;;
    g) GAIN=$OPTARG ;;
    o) OUTDIR=$OPTARG ;;
    n) PREFIX=$OPTARG ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option -$OPTARG"; usage; exit 2 ;;
    :) echo "Option -$OPTARG requires an argument."; usage; exit 2 ;;
  esac
done

# apply defaults for unspecified
FREQ=${FREQ:-$DEFAULT_FREQ}
SR=${SR:-$DEFAULT_SR}
ANT=${ANT:-$DEFAULT_ANT}
LNA=${LNA:-$DEFAULT_LNA}
GAIN=${GAIN:-$DEFAULT_GAIN}
OUTDIR=${OUTDIR:-$DEFAULT_OUTDIR}

# minimal validation
if ! [[ "$ANT" =~ ^[0-9]+$ ]]; then
  echo "Antenna (-a) must be an integer." >&2
  exit 2
fi
if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
  echo "Duration (-t) must be an integer number of seconds." >&2
  exit 2
fi
mkdir -p "$OUTDIR"

# build filename metadata
TS=$(date -u +"%Y%m%dT%H%M%SZ")
HOST="host"
USER=$(whoami)
FNAME="${PREFIX}_${TS}_f${FREQ}_sr${SR}_ant${ANT}_dur${DURATION}s_lna${LNA}_g${GAIN}.s8"
OUTPATH="${OUTDIR%/}/${FNAME}"
META="${OUTPATH%.s8}.json"

# print what will run
cat <<EOF
About to capture:
  output: $OUTPATH
  metadata: $META
  freq: $FREQ Hz
  sample rate: $SR s/s
  antenna: $ANT
  lna: $LNA
  vga gain: $GAIN
  duration: $DURATION s
EOF

# write metadata file (basic, expandable)
cat > "$META" <<JSON
{
  "filename": "$(basename "$OUTPATH")",
  "filepath": "$(realpath "$OUTPATH")",
  "timestamp_utc": "$TS",
  "host": "$HOST",
  "user": "$USER",
  "frequency_hz": $FREQ,
  "sample_rate": $SR,
  "antenna": $ANT,
  "lna": $LNA,
  "vga_gain": $GAIN,
  "duration_seconds": $DURATION,
  "command": "hackrf_transfer -r $(basename "$OUTPATH") -f $FREQ -s $SR -a $ANT -l $LNA -g $GAIN"
}
JSON

# run capture using timeout so it stops after given seconds
# run from OUTDIR to ensure relative filename in metadata/command matches
pushd "$OUTDIR" >/dev/null
if ! command -v hackrf_transfer >/dev/null 2>&1; then
  echo "ERROR: hackrf_transfer not found in PATH." >&2
  popd >/dev/null
  exit 3
fi

# require sudo if not root
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "Not running as root and sudo not available; hackrf_transfer may fail." >&2
  fi
fi

echo "Starting capture (will stop after $DURATION seconds)..."
# use timeout; if timeout not present, fallback to running directly (but user provided duration so we expect timeout)
if command -v timeout >/dev/null 2>&1; then
  $SUDO timeout --preserve-status "${DURATION}s" hackrf_transfer -r "$(basename "$OUTPATH")" -f "$FREQ" -s "$SR" -a "$ANT" -l "$LNA" -g "$GAIN"
  EXITCODE=$?
else
  echo "warning: 'timeout' not found. Running hackrf_transfer without enforced duration." >&2
  $SUDO hackrf_transfer -r "$(basename "$OUTPATH")" -f "$FREQ" -s "$SR" -a "$ANT" -l "$LNA" -g "$GAIN"
  EXITCODE=$?
fi
popd >/dev/null

# finalize metadata: file size, md5 (if available), exit status and finish timestamp
if [ -f "$OUTPATH" ]; then
  FILESIZE=$(stat -c%s "$OUTPATH" 2>/dev/null || stat -f%z "$OUTPATH" 2>/dev/null || echo 0)
else
  FILESIZE=0
fi
if command -v md5sum >/dev/null 2>&1; then
  MD5=$(md5sum "$OUTPATH" | awk '{print $1}' 2>/dev/null || echo null)
elif command -v md5 >/dev/null 2>&1; then
  MD5=$(md5 -q "$OUTPATH" 2>/dev/null || echo null)
else
  MD5=null
fi
FIN_TS=$(date -u +"%Y%m%dT%H%M%SZ")

# append these fields to metadata (simple approach: overwrite with updated JSON)
cat > "$META" <<JSON
{
  "filename": "$(basename "$OUTPATH")",
  "filepath": "$(realpath "$OUTPATH" 2>/dev/null || echo "$OUTPATH")",
  "started_at_utc": "$TS",
  "finished_at_utc": "$FIN_TS",
  "host": "$HOST",
  "user": "$USER",
  "frequency_hz": $FREQ,
  "sample_rate": $SR,
  "antenna": $ANT,
  "lna": $LNA,
  "vga_gain": $GAIN,
  "duration_seconds_requested": $DURATION,
  "actual_exit_status": $EXITCODE,
  "filesize_bytes": $FILESIZE,
  "md5": "$MD5",
  "command": "hackrf_transfer -r $(basename "$OUTPATH") -f $FREQ -s $SR -a $ANT -l $LNA -g $GAIN"
}
JSON

echo "Capture finished. exit=$EXITCODE  file=$OUTPATH  meta=$META"
[ "$EXITCODE" -eq 0 ] || echo "Note: hackrf_transfer returned non-zero ($EXITCODE)." >&2
exit $EXITCODE

