#!/bin/bash
# -*- mode: sh -*-
# Traffic shaping. Code borrowed from https://github.com/awstream/awstream/blob/master/scripts/shaper

set -e
: ${log_file:="fusion.stats"}

function usage() {
    cat <<EOF
Usage:
./scheduled_shape <interface> [schedule]
The schedule is defined as a sequence of [time, bandwidth].
When this script exits, it will automatically clean up the
bandwidth shaping options.
Example:
./scheduled_shape eth0 3000kbit 10 2000kbit 20
EOF
}

function clean() {
    ./shaper.sh clear $1
}

function main() {
    if [ $# -eq 0 ]
    then
       usage
       exit 0
    fi

    iface=$1
    shift

    ## When running this command, we always clean previous specfiications first.
    sleep 1

    trap "clean $iface" SIGHUP SIGINT SIGTERM

    ## has_started will control if we are using `start` or `update`
    has_started=false

    ## iterate all the arguments, (bw, time) pair
    while test $# -gt 1
    do
        bw=$1
        time=$2
        echo "shaping traffic: $bw for $time" | tee ${log_file}

        if [ "$has_started" = true ] ; then
            ./shaper.sh update $iface $bw
        else
            ./shaper.sh start $iface $bw
            has_started=true
        fi

        sleep $time
        shift
        shift
    done

    clean $iface
}

main $@
