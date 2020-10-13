#!/bin/bash

source env.sh

rm -f west.h5
BSTATES="--bstate initial,10"
TSTATES="--tstate bound,-10"
$WEST_ROOT/bin/w_init $BSTATES $TSTATES "$@"
