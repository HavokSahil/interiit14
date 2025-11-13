#!/bin/bash
PIPE1="station.pipe"
PIPE2="bmrep.pipe"
PIPE3="bsstm.pipe"
PIPE4="lmrep.pipe"

foot -e sh -c "cat $PIPE1" &
foot -e sh -c "cat $PIPE2" &
foot -e sh -c "cat $PIPE3" &
foot -e sh -c "cat $PIPE4" &

sudo python3 runner.py
