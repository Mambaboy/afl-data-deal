#include "Python.h"
#include "numpy/arrayobject.h"
#include <iostream>
using namespace std;

void init_numpy(){//初始化 numpy 执行环境，主要是导入包，python2.7用void返回类型，python3.0以上用int返回类型

        import_array();
}

int main()
{
    Py_Initialize();    // 初始化
    if (!Py_IsInitialized()) {
        cout << "Cannot initialize python interface, quitting" << "\n";
        return false;
    }

    // 将Python工作路径切换到待调用模块所在目录，一定要保证路径名的正确性
    string path = "/home/xiaosatianyu/learning";
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
    PyObject* pv = PyObject_GetAttrString(pModule, "main");
    if (!pv || !PyCallable_Check(pv))  // 验证是否加载成功
    {
        cout << "[ERROR] Can't find funftion (test_add)" << endl;
        return 0;
    }
    cout << "[INFO] Get function (test_add) succeed." << endl;

    //设置 numpy格式的参数 PyArrayObject
    //PyObject * args = PyArrayObject();

    // 设置参数
    PyObject* args = PyTuple_New(2);   // 2个参数
    PyObject* arg1 = PyInt_FromLong(4);    // 参数一设为4
    PyObject* arg2 = PyInt_FromLong(3);    // 参数二设为3
    PyTuple_SetItem(args, 0, arg1);
    PyTuple_SetItem(args, 1, arg2);

    // 调用函数
    PyObject* pRet = PyObject_CallObject(pv, args);

    // 获取返回的list参数
    if (pRet)  // 验证是否调用成功
    {
        if (PyList_Check(pRet)){
            int SizeOfList=PyList_Size(pRet);//List对象的大小，这里SizeOfList = 3
			int i;
			for( i = 0; i < SizeOfList; i++){
				PyObject *Item = PyList_GetItem(pRet, i);//获取List对象中的每一个元素
				cout << PyInt_AsLong(Item) <<" "; //输出元素
				Py_DECREF(Item); //释放空间
			 }

            //cout << "result:" << result << "\n";
        }
    }
    cout << "end\n";

    Py_Finalize();      // 释放资源

    return 0;
}

