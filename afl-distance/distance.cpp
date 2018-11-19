
#include "distance.h"
using namespace std;
uint32_t min(uint32_t a, uint32_t b){
	return a <= b ? a : b;
}

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
        son_ids.insert(son_id);
        //m_tree_.insert(std::make_pair(parent_id,son_id));
        Log("new: insert son_id of %d to parent_id of %d\n", son_id,parent_id);
    }
    else{
        m_tree_[parent_id].insert(son_id);
        Log("insert son_id of %d to parent_id of %d\n", son_id,parent_id);
    }
}	

uint32_t Record::GetEditDis(u32 id1, u32 id2){
    
    if (id1==id2)
        return 0;

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
        Log("there is some thing wrong for %s and %s", q1->fname, q2->fname);
        exit(1);
    }    
    if (d1>0 && d1==d2)
        return d1;

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

    uint32_t distance = CalDis ( input1, input2, matrix.row-1, matrix.col-1, matrix);

    //free the heap
    for (i = 0; i < matrix.row; i++){
        free(matrix.content[i]);
        free(matrix.label[i]); 
    }
    free(matrix.content);

    free(input1);
    free(input2);

    Log("the distance between %d and %d is %d\n", q1->id,q2->id,distance);

    // use insert will not overwirte
    // use [] can overwrite
    // update the q1
    if (m_disrecord_.find(id1)==m_disrecord_.end()){
        std::map<u32 ,u32> record;
        record.insert( std::make_pair(id2, distance) );
        m_disrecord_[id1] = record; 		
    }
    else{
        // update the  distance from q2 to q1 
        // here will not overwrite
        m_disrecord_[id1].insert(std::make_pair(id2,distance)); 
    }

    // update the q2
    if (m_disrecord_.find(id2)==m_disrecord_.end()){
        std::map<u32, u32> record;
        record.insert(std::make_pair(id1,distance));
        m_disrecord_[id2] = record; 		
    }
    else{
        // update the distance from q1 to q2
        // here will not overwrite
        m_disrecord_[id2].insert(std::make_pair(id1,distance)); 
    }

    return distance;
}

uint32_t Record::CalDis(u8* input1, u8* input2, uint32_t i, uint32_t j, Matrix matrix_each){

    if (matrix_each.label[i][j]==1)
        return matrix_each.content[i][j];

    //Log("calculte (%d,%d)\n", i,j);
    uint32_t distance;
    if(i == 0){
        distance=j;
    }else if(j == 0){
        distance=i;
    }else{
        distance=
        min(
            min(
                CalDis(input1, input2, i-1, j-1,matrix_each)+(input1[i-1]==input2[j-1] ? 0 : 1),
                CalDis(input1, input2, i, j-1,matrix_each) + 1
               ),
            CalDis(input1, input2, i-1, j,matrix_each) + 1
        );

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
    for( i=0; i < queued_paths; i++){
        for(j=i; j < queued_paths; j++){
            distance = GetEditDis(i,j);
            data[i*queued_paths + j]=distance;
            data[j*queued_paths + i]=distance;
        }
    }

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
    
    callpython(data, queued_paths);
    free(data);
    exit(1);
    //3. print the distance to the python interface

    return 0;
}


uint8_t init_numpy() {
    import_array(); // 需要加入 -fpermissive 编译参数
    return 0;
}
uint8_t callpython(uint32_t* data, u32 inputnum)
{
    Py_Initialize();    // 初始化python 虚拟机
    if (!Py_IsInitialized()) {
        cout << "Cannot initialize python interface, quitting" << "\n";
        return false;
    }
    
    //初始化 numpy 执行环境，主要是导入包，python2.7用void返回类型，python3.0以上用int返回类型
    init_numpy();

    
    // 将Python工作路径切换到待调用模块所在目录，一定要保证路径名的正确性
    string path = "/home/xiaosatianyu/workspace/git/fuzz/afl-data-deal";
    string add_cmd = string("sys.path.append(\"") + path + "\")";
    const char* add_cmd_cstr = add_cmd.c_str();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(add_cmd_cstr);

    // 添加虚拟python路径
    string vpython = "/home/xiaosatianyu/.virtualenvs/xiaosa/local/lib/python2.7/site-packages";
    string vpython_sys = string("sys.path.append(\"") + vpython + "\")";
    const char* vpython_cstr = vpython_sys.c_str();
    PyRun_SimpleString(vpython_cstr);
    
    
    // 加载模块
    PyObject* moduleName = PyString_FromString("plugincluster"); //模块名，不是文件名
    PyObject* pModule = PyImport_Import(moduleName);
    if (!pModule) // 加载模块失败
    {
        cout << "[ERROR] Python get module failed." << endl;
        return 0;
    }
    cout << "[INFO] Python get module succeed." << endl;

    // 加载函数
    PyObject* pv = PyObject_GetAttrString(pModule, "hierarchy_cluster");
    if (!pv || !PyCallable_Check(pv))  // 验证是否加载成功
    {
        cout << "[ERROR] Can't find funftion (test_add)" << endl;
        return 0;
    }
    cout << "[INFO] Get function (test_add) succeed." << endl;
    
    // 参数是一个numpy
    //设置 numpy格式的参数 PyArrayObject
    //uint32_t CArrays[3][3] = {{1, 2, 5}, {4, 7, 8}, {1, 4, 8}};
    //npy_intp Dims[2] = {3, 3};
    
    npy_intp Dims[2] = {inputnum, inputnum};
    
    //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
    PyObject *PyArray  = PyArray_SimpleNewFromData(2, Dims, NPY_UINT32 , data);
    PyObject *ArgArray = PyTuple_New(1);
    PyTuple_SetItem(ArgArray, 0, PyArray); //同样定义大小与Python函数参数个数一致的PyTuple对象

    // 调用函数
    PyObject* pRet = PyObject_CallObject(pv, ArgArray);

    //// 获取返回的list参数
    //if (pRet)  // 验证是否调用成功
    //{
    //    if (PyList_Check(pRet)){
    //        int SizeOfList=PyList_Size(pRet);//List对象的大小，这里SizeOfList = 3
	//		int i;
	//		for( i = 0; i < SizeOfList; i++){
	//			PyObject *Item = PyList_GetItem(pRet, i);//获取List对象中的每一个元素
	//			cout << PyInt_AsLong(Item) <<" "; //输出元素
	//			Py_DECREF(Item); //释放空间
	//		 }

    //        //cout << "result:" << result << "\n";
    //    }
    //}
    
    if (pRet){
        PyArrayObject *ListItem = (PyArrayObject *)(pRet);
        uint32_t Rows = ListItem->dimensions[0], columns = ListItem->dimensions[1];
        for( uint32_t Index_m = 0; Index_m < Rows; Index_m++){
            for(uint32_t Index_n = 0; Index_n < columns; Index_n++){
                //访问数据
                //Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，
                //即可以访问数组元素
                uint32_t x = *(uint32_t *)(ListItem->data + Index_m * ListItem->strides[0] + Index_n * ListItem->strides[1]);
                cout.width(7);
                cout<<x<<" ";
            }
            cout<<endl;
        }

    }

    cout << "end\n";

    Py_Finalize();      // 释放资源

    return 0;
}
