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

# llff iconic data
sh "$SCRIPT" data5_piano

# nerf llff data
sh "$SCRIPT" room
sh "$SCRIPT" horns
sh "$SCRIPT" fortress

# spinnerf dataset
sh "$SCRIPT" 2
sh "$SCRIPT" 3
sh "$SCRIPT" 4
sh "$SCRIPT" 7
sh "$SCRIPT" 10
sh "$SCRIPT" 12
sh "$SCRIPT" book
sh "$SCRIPT" trash