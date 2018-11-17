#ifdef _cplusplus
extern "C" {
#endif

#include "interface.h"

#ifdef _cplusplus
}
#endif

#include "distance.h"

Record* aflRecord = NULL;

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


u32 * GetSelectedSons(u32 parent_id){
    // 1. get all its sons
	u32* selectedids= aflRecord->GetSelectedSons( parent_id);	
    return selectedids; 
}
