#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export | grep HOSTS | ./gen_hostfile.sh
#   export | grep HOSTS | ./gen_hostfile.sh > hostfile

awk '
  BEGIN {
    # print "# Hostfile generated from: export | grep HOSTS"
    # print "# All entries use: user=root"
  }

  {
    line=$0
    sub(/^declare -x /, "", line)

    # Only process *_HOSTS variables
    if (line !~ /_HOSTS=/) next

    split(line, a, "=")
    val=a[2]

    # Remove surrounding quotes
    gsub(/^"/, "", val)
    gsub(/"$/, "", val)

    # Normalize separators
    gsub(/,/, " ", val)
    gsub(/[ \t]+/, " ", val)

    n = split(val, hosts, " ")
    for (i = 1; i <= n; i++) {
      h = hosts[i]
      if (h == "") continue

      printf "%s root\n", h
    }
  }
'
