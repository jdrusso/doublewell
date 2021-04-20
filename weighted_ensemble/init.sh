#!/bin/bash

source env.sh

rm -f west.h5
BSTATES="--bstate initial,1"
TSTATES="--tstate bound,-.00000002"
$WEST_ROOT/bin/w_init $BSTATES $TSTATES "$@"
