#include "interface.h"
#include "distance.h"

Record  *aflRecord = nullptr;

u8 InitDistance( ){
    aflRecord =  new Record();
    if (aflRecord!= NULL)
        return 1;
    else
        return 0;
}

void AddSons(u32 parent_id, u32 son_id){
    aflRecord->AddSons(parent_id, son_id);
}

void UpdateOneDistance(u32 id){
    aflRecord->UpdateOneDistance(id);
}

u32 * GetSelectedSons(u32 parent_id){
    // 1. get all its sons
	u32* selected_ids= aflRecord->GetSelectedSons( parent_id);	
    return selected_ids; 
}


uint32_t GetEditDistance( struct queue_entry * q, u8* mem, u32 len){
    uint32_t distance = aflRecord->GetTargetEditDis(q, mem,len);
    return distance;

}
