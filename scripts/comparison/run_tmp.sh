# Specify a script name to run all scenes in one

# Params
SCRIPT=$1
set -e

# ibrnet data
sh "$SCRIPT" qq3
sh "$SCRIPT" qq6
sh "$SCRIPT" qq10
sh "$SCRIPT" qq11
sh "$SCRIPT" qq13
sh "$SCRIPT" qq16
sh "$SCRIPT" qq17
sh "$SCRIPT" qq21