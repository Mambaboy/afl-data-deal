
#include "distance.h"
uint32_t min(uint32_t a, uint32_t b){
	return a <= b ? a : b;
}

void Log (char const *fmt, ...) {
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
        son_ids.insert(son_id);
        //m_tree_.insert(std::make_pair(parent_id,son_id));
        Log("new: insert son_id of %d to parent_id of %d\n", son_id,parent_id);
    }
    else{
        m_tree_[parent_id].insert(son_id);
        Log("insert son_id of %d to parent_id of %d\n", son_id,parent_id);
    }
}	


uint32_t Record::GetEditDis(u32 id1, u32 id2, uint8_t useold){
    
    //std::cout << id1 << " and " << id2 << std::endl;
    if (id1==id2)
        return 0;
    
    u64 starttime= get_cur_time_us();

    if (useold){ 
        u32 d1,d2;
        d1=d2=0;
        // read from old
        if (m_disrecord_.find(id1)!=m_disrecord_.end())
            if (m_disrecord_[id1].find(id2)!=m_disrecord_[id1].end())
                d1=m_disrecord_[id1][id2]; 		
        
        if (m_disrecord_.find(id2)!=m_disrecord_.end())
            if (m_disrecord_[id2].find(id1)!=m_disrecord_[id2].end())
                d2=m_disrecord_[id2][id1]; 		
        
        if (d1!=d2){
            Log("d1 is %d\n", d1);
            Log("d2 is %d\n", d2);
            Log("there is some thing wrong for %d and %d", id1 ,id2 );
            exit(1);
        }    
        if (d1>0 && d1==d2){
            timeondistance += (get_cur_time_us() - starttime);
            return d1;
        }
    }

    struct queue_entry* q1, *q2;
    u32 i,j;

    q1 = queue;
    i=id1;
    while(i--)
        q1=q1->next;

    q2=queue;        
    j=id2;
    while(j--)
        q2=q2->next;

    Matrix matrix;

    matrix.row = q1->len+1; // [0, a.length] [0,matrix.row)
    matrix.col = q2->len+1;

    u8* input1 = ReadInput(q1->fname, q1->len);
    u8* input2 = ReadInput(q2->fname, q2->len);

    matrix.content = (uint32_t **)malloc( sizeof(uint32_t*) * matrix.row);
    matrix.label   = (uint32_t **)malloc( sizeof(uint32_t*) * matrix.row);

    for (i = 0; i < matrix.row; ++i){
        matrix.content[i] = (uint32_t *) malloc( sizeof(uint32_t) * matrix.col);
        matrix.label[i] = (uint32_t *) calloc(  matrix.col, sizeof(uint32_t));
    }

    uint32_t distance = CalDis ( input1, q1->len, input2, q2->len, matrix.row-1, matrix.col-1, matrix);

    //free the heap
    for (i = 0; i < matrix.row; i++){
        free(matrix.content[i]);
        free(matrix.label[i]); 
    }
    free(matrix.content);

    free(input1);
    free(input2);

    //Log("the distance between %d and %d is %d\n", q1->id,q2->id,distance);

    // use insert will not overwirte
    // use [] can overwrite
    if ( m_disrecord_[id1][id2] != 0) {
        if ( (m_disrecord_[id1][id2] != distance) ||(m_disrecord_[id2][id1] != distance) )
            Log("the distance betwwen %d and %d is from %d to %d\n", id1, id2 ,m_disrecord_[id2][id1], distance );
    }   
    m_disrecord_[id1][id2] = distance;
    m_disrecord_[id2][id1] = distance; // 会覆盖 
    //if (m_disrecord_.find(id1)==m_disrecord_.end()){
    //    std::map<u32 ,u32> record;
    //    record.insert( std::make_pair(id2, distance) ); 
    //    m_disrecord_[id1] = record; 		
    //}
    //else{
    //    m_disrecord_[id1][id2]=distance; // 会覆盖 会新建 
    //}

    //// update the q2
    //if (m_disrecord_.find(id2)==m_disrecord_.end()){
    //    std::map<u32, u32> record;
    //    record.insert(std::make_pair(id1,distance));
    //    m_disrecord_[id2] = record; 		
    //}
    //else{
    //    //m_disrecord_[id2].insert(std::make_pair(id1,distance)); 
    //    m_disrecord_[id2][id1]=distance; // 会覆盖 
    //}
    
    timeondistance += (get_cur_time_us() - starttime);
    return distance;
}

uint32_t Record::CalDis(u8* input1, u32 len1, u8* input2, u32 len2, uint32_t i, uint32_t j, Matrix matrix_each){

    if (matrix_each.label[i][j]==1)
        return matrix_each.content[i][j];

    //Log("calculte (%d,%d)\n", i,j);
    uint32_t distance;
    if(i == 0){
        distance=j;
    }else if(j == 0){
        distance=i;
    }else{
        if  ( (len1/len2 > 5 || len2/len1 >5) &&(len1 > 1000 || len2 > 1000) ){
            if (len1> len2)
                distance =len1;
            else
                distance= len2;
        }
        else{
            distance=
            min(
                min(
                    CalDis(input1,len1, input2,len2, i-1, j-1,matrix_each)+(input1[i-1]==input2[j-1] ? 0 : 1),
                    CalDis(input1,len1, input2,len2, i, j-1,matrix_each) + 1
                   ),
                CalDis(input1, len1, input2,len2, i-1, j,matrix_each) + 1
            );
        }

    }
    matrix_each.content[i][j]=distance;
    matrix_each.label[i][j]=1;
    return distance;
}

u8* Record::ReadInput(u8* fname, u32 len){
    u8 fd;
    u8* mem;
    fd = open((char*)fname, O_RDONLY);
    if (fd < 0) PFATAL("Unable to open '%s'", fname);

    mem = (u8*)calloc(sizeof(u8),len);

    if (read(fd, mem, len) != len)
        FATAL("Short read from '%s'", fname);

    close(fd);

    return mem;
}
void Record::UpdateOneDistance(u32 id){
    uint32_t i;
    uint32_t distance;
    for( i=0; i < queued_paths; i++){
        distance = GetEditDis(i, id, 0);
    }    

}

uint32_t* Record::GetSelectedSons(u32 parent_id){
    //1. get the sons id
    //std::set<u32> sonids;
    //if (m_tree_.find(parent_id) == m_tree_.end())
    //    return (u32*)0;
    //else{
    //    sonids = m_tree_[parent_id];
    //}

    //2. calculate the distance between each other in the sons
    // get the distance matrix
    //for (auto id1 : sonids){
    //    for (auto id2: sonids){
    //        GetEditDis(id1, id2);
    //    }
    //}

    uint32_t i, j, distance;
    uint32_t *data=(uint32_t*) malloc(queued_paths * queued_paths * sizeof(uint32_t));

    timeondistance=0;
    u8 buffer [50];
    u8 threadnum=1; // 1 或者2
    std::queue< std::future<uint32_t> > workers;
    for( i=0; i < queued_paths; i++){
        for(j=i; j < queued_paths-threadnum+1; j=j+threadnum){
            if (threadnum-1){
                std::future<uint32_t> getdistance0 = std::async(std::launch::async,&Record::GetEditDis, this , i, j+0 ,1 );
                std::future<uint32_t> getdistance1 = std::async(std::launch::async,&Record::GetEditDis, this , i, j+1 ,1 );
                std::future<uint32_t> getdistance2 = std::async(std::launch::async,&Record::GetEditDis, this , i, j+2 ,1);
                std::future<uint32_t> getdistance3 = std::async(std::launch::async,&Record::GetEditDis, this , i, j+3 ,1);
                std::future<uint32_t> getdistance4 = std::async(std::launch::async,&Record::GetEditDis, this , i, j+4 ,1);
               
                uint32_t distance0 = getdistance0.get();
                uint32_t distance1 = getdistance1.get();
                uint32_t distance2 = getdistance2.get();
                uint32_t distance3 = getdistance3.get();
                uint32_t distance4 = getdistance4.get();
                data[ i*queued_paths     + (j+0)] = distance0;
                data[ (j+0)*queued_paths +  i ] = distance0;
                
                data[ i*queued_paths     + (j+1)] = distance1;
                data[ (j+1)*queued_paths +  i ] = distance1;
                
                data[ i*queued_paths     + (j+2)] = distance2;
                data[ (j+2)*queued_paths +  i ] = distance2;
                
                data[ i*queued_paths     + (j+3)] = distance3;
                data[ (j+3)*queued_paths +  i ] = distance3;
                
                data[ i*queued_paths     + (j+4)] = distance4;
                data[ (j+4)*queued_paths +  i ] = distance4;

            }
            else{
                distance = GetEditDis(i,j,1);
                data[ i*queued_paths + j] = distance;
                data[ j*queued_paths + i] = distance;
            }
            
            if (stop_soon)
               exit(0);

            //for (uint8_t m =0; m < threadnum; m++){
            //    //std::future<uint32_t> getdistance( std::async(std::launch::async,&Record::GetEditDis, this , i, j+m) );
            //    std::future<uint32_t> getdistance = std::async(std::launch::async,&Record::GetEditDis, this , i, j+m );
            //    workers.push(getdistance);
            //}
           
            ////for (uint8_t m =0; m < threadnum; m++){
            //    //std::future<uint32_t> getdistance = workers.pop(); 
            //    //uint32_t dtemp = getdistance.get();
            //    //distance =dtemp;
            //    distance = GetEditDis(i,j);
            //    //if (distance != dtemp)
            //    //    exit(0);
            //    data[ i*queued_paths     + (j+m)] = distance;
            //    data[ (j+m)*queued_paths +  i ] = distance;
            //    
            //    if (stop_soon)
            //        exit(0);
            ////}
        }
        // set   stage name
        sprintf (buffer, "calculate-%d",i);
        stage_name= buffer;
        show_stats();
    }
    // for test distance
    //uint32_t d1,d2;
    //Log("\n");
    //for( i=0; i < queued_paths; i++){
    //    for(j=i; j < queued_paths; j++){
    //        d1 = data[i*queued_paths + j];
    //        d2 = data[j*queued_paths + i];
    //        Log("the distance between %d and %d is %d and %d\n", i, j,d1,d2);
    //    }
    //}
    //exit(1);
    uint32_t * result; 
    result = CallPython(data, queued_paths);
    free(data);
    //3. print the distance to the python interface
    Log("in this process, cost %llu time on distancd calcualtion", timeondistance );
    return result;
}


uint32_t Add(uint32_t x){
    return (x+1);
}

uint8_t InitNumpy() {
    import_array(); // 需要加入 -fpermissive 编译参数
    return 0;
}

void InitPython(){
    if (initpython==1)
       return;
    Py_Initialize();    // 初始化python 虚拟机
    if (!Py_IsInitialized()) {
        Log("Cannot initialize python interface, quitting\n");
        std::cout << "Cannot initialize python interface, quitting" << "\n";
        exit(1);
    }
    
    //初始化 numpy 执行环境，主要是导入包，python2.7用void返回类型，python3.0以上用int返回类型
    InitNumpy();
    
    // 将Python工作路径切换到待调用模块所在目录，一定要保证路径名的正确性
    std::string path = "/home/xiaosatianyu/workspace/git/fuzz/afl-data-deal";
    std::string add_cmd = std::string("sys.path.append(\"") + path + "\")";
    const char* add_cmd_cstr = add_cmd.c_str();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(add_cmd_cstr);
    
    // 将Python工作路径切换到待调用模块所在目录，一定要保证路径名的正确性
    std::string path2 = "/home/xiaosatianyu/learning";
    std::string add_cmd2 = std::string("sys.path.append(\"") + path2 + "\")";
    const char* add_cmd_cstr2 = add_cmd2.c_str();
    PyRun_SimpleString(add_cmd_cstr2);

    // 添加虚拟python路径
    std::string vpython = "/home/xiaosatianyu/.virtualenvs/xiaosa/local/lib/python2.7/site-packages";
    std::string vpython_sys = std::string("sys.path.append(\"") + vpython + "\")";
    const char* vpython_cstr = vpython_sys.c_str();
    PyRun_SimpleString(vpython_cstr);
    
    // 加载模块
    PyObject* moduleName = PyString_FromString("plugincluster"); //模块名，不是文件名
    PyObject* pModule = PyImport_Import(moduleName);
    if (!pModule) // 加载模块失败
    {
        Log("[ERROR] Python get module failed.\n");
        std::cout << "[ERROR] Python get module failed." << std::endl;
        exit(1);
    }
    std::cout << "[INFO] Python get module succeed." << std::endl;
    Log("[INFO] Python get module succeed.\n");

    // 加载函数
    pv = PyObject_GetAttrString(pModule, "getcluster");
    if (!pv || !PyCallable_Check(pv))  // 验证是否加载成功
    {
        Log("[ERROR] Can't find funftion (test_add)\n");
        std::cout << "[ERROR] Can't find funftion (test_add)" << std::endl;
        exit(1);
    }
    std::cout << "[INFO] Get function (test_add) succeed." << std::endl;
    Log( "[INFO] Get function (test_add) succeed.\n");
   
    // init ok
    initpython=1;

}


uint32_t * CallPython(uint32_t* data, u32 inputnum){
    InitPython(); 
    npy_intp Dims[2] = {inputnum, inputnum};
    //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，
    //第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
    PyObject *PyArray  = PyArray_SimpleNewFromData(2, Dims, NPY_UINT32 , data);
    PyObject *ArgArray = PyTuple_New(1);
    PyTuple_SetItem(ArgArray, 0, PyArray); //同样定义大小与Python函数参数个数一致的PyTuple对象

    // 调用函数
    PyObject* pRet = PyObject_CallObject(pv, ArgArray);
    
    // 返回的是一个一维的 聚类结果
    uint32_t * selected_ids;
    uint32_t num;
    PyArrayObject *result;
    if (pRet){
        result = (PyArrayObject *)(pRet);
        num = result->dimensions[0];
        Log("\nselected %d inputs\n",num);
        selected_ids =(uint32_t*) malloc( (num+1)*sizeof(uint32_t)); // free in the later
        for( uint32_t m = 0; m < num; m++){
            //访问数据
            //m 和 n 分别是数组元素的坐标，乘上相应维度的步长，
            uint32_t x = *(uint32_t *)(result->data + m * result->strides[0]);
            selected_ids[m]=x;
            if (x> queued_paths)
                sleep(100000);
            Log("return a selected input of %d\n", x);
            //std::cout.width(7);
            //std::cout<<x<<" ";
        }
    }
    std::cout << "end\n";
    //Py_Finalize();      // 释放资源
    selected_ids[num]=(uint32_t)(-1); 
    return selected_ids;
}
