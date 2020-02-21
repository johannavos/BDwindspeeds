#! /bin/bash

# Set up a directory on a work machine to do the VLA data processing. The
# current directory should be the top of the processing tree, with downloaded
# SDM files contained in a subdirectory called "sdm". The path to the Git repo
# is found by looking at the path to this script. It should be possible to run
# this script repeatedly idempotently.

[ -e repo ] || ln -sf $(dirname $0) repo

while read nick sdm ; do
    mkdir -p $nick
    ln -sf ../sdm/$sdm $nick/raw.sdm
    ln -sf ../repo/process.py $nick/process.py
    echo $nick >$nick/.rxpackage_ident
done <<EOF
day1  18A-427.sb35495301.eb35641262.58403.43754182871
day2  18A-427.sb35495301.eb35642928.58404.44638675926
day3  18A-427.sb35495301.eb35642944.58405.43316700232
EOF
