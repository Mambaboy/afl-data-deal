file BIN
set verbose off
set complaints 0
set logging file /tmp/tmp_gdb.txt
set logging redirect on
set logging overwrite on
set logging on
run PARA CRASH
printf "[+] Backtrace:\n"
bt 1
set logging redirect off
set logging off
q

