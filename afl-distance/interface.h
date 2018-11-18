#ifndef INTERFACE_H
#define INTERFACE_H

#include "afl-fuzz.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void AddSons(u32 parent_id, u32 son_id);

// get the selected id of sons
// return a dynamic  u32[], the last is 0x00
//u32* GetSelectedSons(u32 parent_id);


// init a glboal interface to calculate  cache and get the distance 
// Return: 1 if intilized successfully, otherwise 0.
u8 InitDistance();

#ifdef __cplusplus
}
#endif

#endif
