#ifndef DISTANCE_h
#define DISTANCE_H


#ifdef _cplusplus
extern "C" {
#endif
#include "afl-fuzz.h"
#include "interface.h"
#ifdef _cplusplus
}
#endif

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdarg.h>
#include <set>
#include <map>


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

uint32_t min(uint32_t a, uint32_t b){
	return a <= b ? a : b;
}

class Record{
    public:

		u32 m_inputs_num_; 
		Tree m_tree_; // record the relationship between parent and  sons
		Distance_record m_disrecord_;  // cache all the distance between inputs

		Record();
		~Record();
	    uint32_t * GetSelectedSons(u32 parent_id);	
		void AddSons(u32 parent_id, u32 son_id);
        void Log (char const *fmt, ...);
		uint32_t GetEditDis(struct queue_entry q1, struct queue_entry q2);
		uint32_t CalDis(u8* input1, u8* input2, uint32_t i, uint32_t j, Matrix matrix_each);
		u8* ReadInput(u8* fname, u32 len);

};




#endif // end DISTANCE_H


