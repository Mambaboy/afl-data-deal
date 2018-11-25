Outdir=/dev/shm/afl-orig
rm -rf $Outdir
gdb --args ./afl-orig/afl-fuzz -d  -i ./seed  -o $Outdir ./nm -C @@  
