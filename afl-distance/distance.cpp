#ifndef DISTANCE_H_
#define DISTANCE_H_

#include "distance.h"

void Record::Log (char const *fmt, ...) {
    static FILE *f = NULL;
    char logfile[512];
    memset(logfile, 0, 512);

    sprintf(logfile, "%s/distance.log", out_dir);
    if (f == NULL) {
      f= fopen(logfile, "w");
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fflush(f);
}

void Record::AddSons(u32 parent_id, u32 son_id){
    if (m_tree_.find(parent_id)==m_tree_.end()){
        std::set<u32> son_ids;
        son_ids.insert(son);
        //m_tree_.insert(std::make_pair(parent_id,son_id));
        Log("new: insert son_id of %s to parent_id of %s\n", son,parent);
    }
    else{
        m_tree_[parent_id].insert(son_id);
        Log("insert son_id of %s to parent_id of %s\n", son,parent);
    }
}	



uint32_t Record::GetEditDis(struct queue_entry* q1, struct queue_entry* q2){

    u32 d1,d2;
	d1=d2=0;
    
	// read from old
    if (m_disrecord_.find(q1->id)!=m_disrecord_.end()){
        d1=m_disrecord_[q1->id][q2->id]; 		
    }
    if (m_disrecord_.find(q2->id)==m_disrecord_.end()){
        d2=m_disrecord_[q2->id][q1->id]; 		
    }
    if (d1!=d2)
        Log("there is some thing wrong for %s and %s", q1->fname, q2->fname);
    if (d1>0 && d1=d2)
        return d1;

    Matrix matrix;
    uint32_t i,j;

    matrix.row = q1->len+1; // [0, a.length] [0,matrix.row)
    matrix.col = q2->len+1;

    u8* input1 = readinput(q1->fname, q1->len);
    u8* input2 = readinput(q2->fname, q2->len);

    matrix.content = (uint32_t **)malloc( sizeof(uint32_t*) * matrix.row);
    matrix.label   = (uint32_t **)malloc( sizeof(uint32_t*) * matrix.row);

    for (i = 0; i < matrix.row; ++i){
        matrix.content[i] = (uint32_t *) malloc( sizeof(uint32_t) * matrix.col);
        matrix.label[i] = (uint32_t *) calloc(  matrix.col, sizeof(uint32_t));
    }

    uint32_t distance= CalDis( input1, input2, matrix.row-1, matrix.col-1, matrix);

    //free the heap
    for (i = 0; i < matrix.row; i++){
        free(matrix.content[i]);
        free(matrix.label[i]);
    }
    free(matrix.content);

    free(input1);
    free(input2);

    Log("the distance between %s and %s is %d\n", q1->fname,q2->fname,distance);

    // use insert will not overwirte
    // use [] can overwrite
    // update the q1
    if (m_disrecord_.find(q1->id)==m_disrecord_.end()){
        std::map<u32.u32> record;
        record.insert(std::makr_pari<u32,u32>(q2->id,distance));
        m_disrecord_[q1->id] = record; 		
    }
    else{
        // update the  distance from q2 to q1 
        // here will not overwrite
        m_disrecord_[q1->id].insert(std::makr_par<u32,u32>(q2->id,distance))  
    }

    // update the q2
    if (m_disrecord_.find(q2->id)==m_disrecord_.end()){
        std::map<u32.u32> record;
        record.insert(std::makr_pari<u32,u32>(q1->id,distance));
        m_disrecord_[q2->id] = record; 		
    }
    else{
        // update the distance from q1 to q2
        // here will not overwrite
        m_disrecord_[q2->len].insert(std::makr_par<u32,u32>(q1->len,distance))  
    }

    return distance;
}

uint32_t Record::CalDis(u8* input1, u8* input2, uint32_t i, uint32_t j, Matrix matrix_each){

    if (matrix_each.label[i][j]==1)
        return matrix_each.content[i][j];

    Log("calculte (%d,%d)\n", i,j);
    uint32_t distance;
    if(i == 0){
        distance=j;
    }else if(j == 0){
        distance=i;
    }else{
        distance=min(
                    min(
                        dis(input1, input2, i-1, j-1,matrix_each)+(input1[i-1]==input2[j-1] ? 0 : 1),
                        dis(input1, input2, i, j-1,matrix_each) + 1
                       ),
                    dis(input1, input2, i-1, j,matrix_each) + 1
                );

    }
    matrix_each.content[i][j]=distance;
    matrix_each.label[i][j]=1;
    return distance;
}

u8* Record::ReadInput(u8* fname, u32 len){
    u8 fd;
    u8* mem;
    fd = open(fname, O_RDONLY);
    if (fd < 0) PFATAL("Unable to open '%s'", q->fname);

    mem = ck_alloc_nozero(len);

    if (read(fd, mem, len) != len)
      FATAL("Short read from '%s'", fname);

    close(fd);

    return mem;
}

uint32_t* Record::GetSelectedSons(u32 parent_id){
	//1. get the sons id
	std::set<u32> sonids;
	if (m_tree_.find(parent_id) == m_tree_.end())
		return (u32*)0;
	else{
		sonids = m_tree_[parent];
	}

	//2. calculate the distance between each other in the sons
 	u32 id1,id2;




}





#endif /* ^ DISTANCE_H_ */