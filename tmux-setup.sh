#!/bin/bash

SESSIONNAME="{{SESSIONNAME}}"
tmux has-session -t $SESSIONNAME 2> /dev/null

if [ $? != 0 ]
  then
    # new session with name $SESSIONNAME and window 0 named "base"
    tmux new-session -s $SESSIONNAME -n base -d

    tmux new-window -t $SESSIONNAME:1 -n "deploy"
    tmux send-keys 'source ~/.aws.profile' 'C-m'
    tmux send-keys 'cd data-infra-deploy' 'C-m'

    tmux new-window -t $SESSIONNAME:2 -n "hive"
    tmux send-keys 'hive' 'C-m'

    tmux new-window -t $SESSIONNAME:3 -n "presto"
    tmux send-keys 'presto' 'C-m'

    # switch the "base" window
    tmux select-window -t $SESSIONNAME:0
fi

tmux attach
