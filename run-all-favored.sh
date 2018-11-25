Outdir=/dev/shm/afl-all-favored
rm -rf $Outdir
 ./afl-all-favored/afl-fuzz -d  -i ./seed  -o $Outdir ./nm -C @@  
