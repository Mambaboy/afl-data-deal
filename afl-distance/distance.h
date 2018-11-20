#ifndef DISTANCE_H
#define DISTANCE_H


#ifdef __cplusplus
extern "C" {
#endif

#include "afl-fuzz.h"
#include "interface.h"

#ifdef __cplusplus
}
#endif

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdarg.h>
#include <set>
#include <map>
#include <sys/stat.h> 

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include  <python2.7/Python.h>
#include <numpy/arrayobject.h>

// the matrix_each for each distance betwwen two input
struct Matrix{
    uint32_t row;
    uint32_t col;
    uint32_t **content;
    uint32_t **label;
};

// the first is the id of parent node, the second is ids the son nodes 
typedef std::map< u32, std::set<u32> > Tree; 


//the first is the id of current input, the second is a pair, 
//in which the first is the other input, and the distance beween each other
// use id can quickly find the entry in queue
typedef std::map<u32, std::map<u32,u32> > Distance_record;	

class Record{
    public:
        u32 m_inputs_num_; 
	    Tree m_tree_; // record the relationship between parent and  sons
	    Distance_record m_disrecord_;  // cache all the distance between inputs

	    Record(){};
	    ~Record(){};
        uint32_t * GetSelectedSons(u32 parent_id);	
        void AddSons(u32 parent_id, u32 son_id);
        uint32_t GetEditDis(uint32_t id1, uint32_t id2);
        uint32_t CalDis(u8* input1, u8* input2, uint32_t i, uint32_t j, Matrix matrix_each);
        u8* ReadInput(u8* fname, u32 len);

};

// 函数
static void Log (char const *fmt, ...);
uint8_t InitNumpy();
uint32_t *CallPython(uint32_t * data, u32 inputnum);
void InitPython();

//变量
static uint8_t initpython=0;
static PyObject* pv; // the target fuction


// 外部的函数和变量
extern void show_stats(void);

extern u8* out_dir;
extern struct queue_entry * queue;
extern u32 queued_paths;
extern u8* stage_name; 
extern u8 stop_soon;

#endif // end DISTANCE_H


