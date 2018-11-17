#include "interface.h"
#include "distance.h"

Record* aflRecord = NULL;

u8 InitDistance(){
    aflRecord =  new Record();
    if (aflRecord!= NULL)
        return 1;
    else
        return 0;
}

void AddSon(u32 parent_id, u32 son_id){
    aflRecord.AddSon(parent_id, son_id);
}


u32 * GetSelectedSons(u32 parent_id){
    // 1. get all its sons
	u32* selectedids= aflRecord.GetSelectedSons(u32 parent_id);	
    return selectedids; 
}
