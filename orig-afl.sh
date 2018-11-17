rm -rf /dev/shm/afl-with-trim
/home/xiaosatianyu/learning/afl-orig/afl-fuzz  -i /home/xiaosatianyu/test/seed  -o /dev/shm/afl-with-trim /home/xiaosatianyu/test/install/bin/nm -C @@  
