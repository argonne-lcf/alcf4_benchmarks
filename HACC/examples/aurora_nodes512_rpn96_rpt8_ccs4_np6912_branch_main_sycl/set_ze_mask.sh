#!/bin/bash

rpn=${RANKS_PER_NODE}
ranks_per_tile=${RANKS_PER_TILE}
verbose=${VERBOSE_RANK_NODE_MAP}

##### 0.0 0.1 0.0 0.1 until full 1.0 1.1 1.0 1.0 until full
##### My GPU first, tiles alternate
((g=PALS_LOCAL_RANKID/2/ranks_per_tile))
((t=PALS_LOCAL_RANKID%2))
export ZE_AFFINITY_MASK=$g.$t

######### Alternative placement: 0.0 0.0 0.0 0.0 .... 0.1 0.1 0.1 0.1 ....
#g=0
#t=0
#counter=0
#for ((r=0; r<${rpn}; r++)); do
#
#   if ((PALS_LOCAL_RANKID==r)); then
#       export ZE_AFFINITY_MASK=$g.$t
#   fi    
#   
#   #echo rank=${r} ZE_AFFINITY_MASK=${g}.${t}
#
#   ((counter++))
#   if ((counter==ranks_per_tile)); then
#      counter=0
#      ((t++))
#   fi
#
#   if ((t==2)); then
#       ((g++))
#       t=0
#   fi
#done

if [[ "$verbose" =~ ^(true|1)$ ]]; then
  echo "[I am rank $PALS_RANKID on node $(hostname)] Localrank=$PALS_LOCAL_RANKID, ZE_AFFINITY_MASK=$ZE_AFFINITY_MASK" >> RANK_NODE_MAP.log
fi

# Launch the executable:
$*
