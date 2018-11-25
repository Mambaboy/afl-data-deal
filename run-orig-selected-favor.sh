Outdir=/dev/shm/afl-orig-selected-favor
rm -rf $Outdir
./afl-orig/afl-fuzz -d  -i ./seed  -o $Outdir ./nm -C @@  
