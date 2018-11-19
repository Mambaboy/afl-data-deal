#include "Python.h"
#include <numpy/arrayobject.h>
#include <iostream>
using namespace std;
int init_numpy() {
    import_array(); // 需要加入 -fpermissive 编译参数
    return 0;
}
int main()
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
    PyObject* moduleName = PyString_FromString("fuck"); //模块名，不是文件名
    PyObject* pModule = PyImport_Import(moduleName);
    if (!pModule) // 加载模块失败
    {
        cout << "[ERROR] Python get module failed." << endl;
        return 0;
    }
    cout << "[INFO] Python get module succeed." << endl;

    // 加载函数
    PyObject* pv = PyObject_GetAttrString(pModule, "fuck");
    if (!pv || !PyCallable_Check(pv))  // 验证是否加载成功
    {
        cout << "[ERROR] Can't find funftion (test_add)" << endl;
        return 0;
    }
    cout << "[INFO] Get function (test_add) succeed." << endl;
    
    // 参数是一个numpy
    //设置 numpy格式的参数 PyArrayObject
    double CArrays[3][3] = {{1.3, 2.4, 5.6}, {4.5, 7.8, 8.9}, {1.7, 0.4, 0.8}};
    npy_intp Dims[2] = {3, 3};
    
    //生成包含这个多维数组的PyObject对象，使用PyArray_SimpleNewFromData函数，第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组
    PyObject *PyArray  = PyArray_SimpleNewFromData(2, Dims, NPY_DOUBLE, CArrays);
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
        int Rows = ListItem->dimensions[0], columns = ListItem->dimensions[1];
		for( int Index_m = 0; Index_m < Rows; Index_m++){
            for(int Index_n = 0; Index_n < columns; Index_n++){
				//访问数据
                //Index_m 和 Index_n 分别是数组元素的坐标，乘上相应维度的步长，
				//即可以访问数组元素
                double x = *(double *)(ListItem->data + Index_m * ListItem->strides[0] + Index_n * ListItem->strides[1]);
                cout<<x<<" ";
            }
            cout<<endl;
        }

    }

    cout << "end\n";

    Py_Finalize();      // 释放资源

    return 0;
}

