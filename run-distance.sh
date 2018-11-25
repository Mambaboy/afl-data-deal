Ourdir=/dev/shm/afl-distance
rm -rf $Ourdir
gdb --args  ./afl-distance/afl-fuzz -d  -i ./seed  -o $Ourdir ./nm -C @@  
