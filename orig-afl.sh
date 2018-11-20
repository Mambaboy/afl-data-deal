rm -rf /dev/shm/afl-orig
gdb --args ./afl-orig/afl-fuzz  -i ./seed  -o /dev/shm/afl-orig ./nm -C @@  
