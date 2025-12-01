#!/bin/bash
sudo python3 runner.py > "run_$(date +%s).log"
rm *.sock
