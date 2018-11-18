rm -rf /dev/shm/afl-distance
gdb --args ./afl-distance/afl-fuzz -d  -i ./seed  -o /dev/shm/afl-distance ./nm -C @@  
