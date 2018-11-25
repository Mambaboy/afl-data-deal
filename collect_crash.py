#!/usr/bin/env python
#coding=utf-8

import logging
import coloredlogs
#创建记录器
l=logging.getLogger("crash")
l.setLevel("INFO") #记录器的级别
fmt = "%(asctime)-15s %(filename)s:%(lineno)d %(process)d %(levelname)s %(message)s"
coloredlogs.install(fmt=fmt, logger=l)


import os
import re
import sys
import time
import shutil
import subprocess
    

class Crashes(object):
    def __init__(self, crash_dir, binary_path, crash_store_dir, para):  

        # basic info
        self.binary_path = binary_path
        self.crash_dir = crash_dir
        
        # crash dir
        self.crash_store_dir = crash_store_dir
        if not os.path.exists(self.crash_store_dir):
            os.makedirs(self.crash_store_dir)  
        
        self.para = para

        self.crash_block_set=set() # the crash address


        self.template_contents = None
        self._load_gdb_template()

    #-----------------GDB Module--------------------------------------------
    def _load_gdb_template(self):
        # load the templet shell script
        # if no script found, raise an excption
        # the loaction of templet is at the same directory default as this .py file 
        pwd = os.path.dirname(os.path.realpath(__file__)) #os.getcwd()
        template_filename = "gdb_script_template.gdb"
        template_path = os.path.join(pwd, template_filename)
        try:
            with open(template_path) as f_obj:
                template_contents = f_obj.read()
        except IOError as e:
            l.warn(e)
        self.template_contents = template_contents

    # 生成一个针对一个crash的gdb模板
    def gdb_script_crafter(self, crash_path):
        # generate the gdb script
        bin_pattern = 'BIN'
        crash_pattern = 'CRASH'
        para_pattern = 'PARA'

        out_shell1 = re.sub(bin_pattern, self.binary_path, self.template_contents)
        out_shell2 = re.sub(crash_pattern, crash_path, out_shell1)
        out_shell3 = re.sub(para_pattern, self.para, out_shell2)

        # the crafted gdb script is in the same directionary as this: script
        gdb_script_file_name = "/tmp/gdb_script"
        try:
            with open(gdb_script_file_name, 'w') as f:
                f.write(out_shell3)
        except IOError as e:
            l.error("fail to open %s",gdb_script_file_name)
            l.warn(e)

        return gdb_script_file_name


    def get_crash_info(self, crash_path):
        sig_type = 0
        bt_addr =''

        #pedb 和original gdb 还不一样
        # 记录结果的文件
        gdblog_filename = '/tmp/tmp_gdb.txt'
        gdblog_path = os.path.join(os.getcwd(), gdblog_filename)
        with open(gdblog_path, 'w') as f:
            pass # 先生产一下文件
        # use RE to findout the latest bt in the GDB DEBUG log
        gdb_script_name = self.gdb_script_crafter(crash_path)
        shcmd = 'gdb --tty=/dev/null -x '+ gdb_script_name + ' --batch --quiet &>> /dev/null'
        p = os.popen(shcmd) #堵塞的
        gdblog_contents=''
        times = 300
        while times:
            try:
                with open(gdblog_path, 'r') as f_obj:
                    gdblog_contents = f_obj.read()
            except IOError as e:
                pass
            if "Backtrace" in gdblog_contents:
                break
            time.sleep(0.01)
            times -= 1
        
        if len(gdblog_contents) ==0:
            l.warn("%s fail to get gdb result", os.path.basename(crash_path))
            return (sig_type, bt_addr)

        # use RE to get the latest backtrack
        bt_pattern = re.compile(r'(0x.*?) in ')
        bt_result = bt_pattern.findall(gdblog_contents) 
        if bt_result:
            bt_addr = bt_result[0]
        else:
            l.warn('%s fail to get the crash addr, use defualt', crash_path )
            return (sig_type, bt_addr)
        
        # use RE to get the error signal
        sigill_pattern = re.compile(r'SIGILL')
        sigsegv_pattern = re.compile(r'SIGSEGV')
        sigabrt_pattern = re.compile(r'SIGABRT')

        sigill_result = sigill_pattern.findall(gdblog_contents)
        sigsegv_result = sigsegv_pattern.findall(gdblog_contents) 
        sigabrt_result = sigabrt_pattern.findall(gdblog_contents) 

        if sigill_result:
            sig_type = 4
        if sigsegv_result:
            sig_type = 11
        if sigabrt_result:
            sig_type = 6

        return (sig_type, bt_addr)

    
    #---------------------------------------------------------------------------------
    #deal with each testcase
    def deal_tc(self, tc_path):
        
        tc_signal, crash_address = self.get_crash_info(tc_path)
        tc_signal = int(tc_signal)
        crash_address = str(crash_address)
        tc_name = os.path.basename(tc_path)
        
        if tc_signal == 0:
            return #表示没有崩溃, or uninteresting SIG
        l.info("%s get signal %d, crash at %s", os.path.basename(tc_path), tc_signal, crash_address)
        
        #如果不能收集奔溃点
        if crash_address == 0:
            tag="no_address"
            new_path = os.path.join(self.crash_store_dir, str(tc_signal), crash_address,  tc_name)
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path)) #创建多层目录 
            
            if os.path.exists(new_path): #是否已经存在了   
                #l.info("%s has exists",new_path)
                return
            self.crash_block_set.add(crash_address) #记录的是崩溃处的地址
            
            shutil.copyfile(tc_path, new_path) #copy to the tmp dir
        
        #如果可以收集崩溃点，且是新的
        elif not crash_address in self.crash_block_set:
            new_path= os.path.join(self.crash_store_dir, str(tc_signal), crash_address, tc_name) #
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path)) #创建多层目录 
            
            if os.path.exists(new_path): #是否已经存在了   
                #l.info("%s has exists",new_path)
                return
            self.crash_block_set.add(crash_address) #记录的是崩溃处的地址
            
            shutil.copyfile(tc_path, new_path) #copy to the tmp dir 
                    
        #如果可以收集崩溃点，但是重复的   
        else:
            tag="redundant"
            new_path = os.path.join(self.crash_store_dir, tag , str(tc_signal), tc_name) #重命名
            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path)) #创建多层目录
            # 如果已经收集过了,既有对应文件了
            if os.path.exists(new_path) :
                #l.info("%s has exists",new_path)
                return
            
            shutil.copyfile(tc_path, new_path) #copy to the tmp dir

    #遍历crash
    def go_throught(self):
        for tc in sorted(os.listdir(self.crash_dir)):
            if "README" in tc :
                continue
            #处理
            tc_path = os.path.join(self.crash_dir,tc)
            self.deal_tc(tc_path)
            
 
#----------------------------------------------------------------------
def main():
    """"""
    l.info("start to collect crash")
    
    crash_dir = "/dev/shm/afl-distance/crashes"
    binary_path = "/home/xiaosatianyu/learning/nm"
    crash_store_dir = "/home/xiaosatianyu/learning/poc-distance" 
    para =" -C " 
    Collect_crashes=Crashes(crash_dir, binary_path, crash_store_dir, para= para)
    Collect_crashes.go_throught()

if __name__ == "__main__":
    sys.exit(main())

