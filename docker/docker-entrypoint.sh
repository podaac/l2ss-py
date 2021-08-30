#!/bin/bash
set -e

if [ "$1" = 'l2ss-py' ]; then
  exec l2ss-py "$@"
elif [ "$1" = 'l2ss_harmony' ]; then
  exec l2ss_harmony "$@"
else
  exec l2ss_harmony "$@"
fi
