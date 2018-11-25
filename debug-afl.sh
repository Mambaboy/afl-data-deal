Outdir=/dev/shm/afl-distance-debug

rm -rf $Outdir
gdb --args  ./afl-distance/afl-fuzz -d  -i ./seed  -o $Outdir ./nm -C @@  
