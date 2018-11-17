#ifndef INTERFACE_H
#define INTERFACE_H

#ifdef _cplusplus
extern "C" {
#endif

#include "afl-fuzz.h"

// init a glboal interface to calculate  cache and get the distance 
// Return: 1 if intilized successfully, otherwise 0.
u8 InitDistance();

void AddSon(u32 parent_id, u32 son_id);

// get the selected id of sons
// return a dynamic  u32[], the last is 0x00
u32* GetSelectedSons(u32 parent_id);

#ifdef _cplusplus
}
#endif

#endif
