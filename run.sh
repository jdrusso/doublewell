#!/bin/bash

source env.sh

rm -f west.log
$WEST_ROOT/bin/w_run "$@" 2>&1 | tee -a west.log
