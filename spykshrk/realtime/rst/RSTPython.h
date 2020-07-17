#ifndef __PYX_HAVE__RSTPython
#define __PYX_HAVE__RSTPython


#ifndef __PYX_HAVE_API__RSTPython

#ifndef __PYX_EXTERN_C
  #ifdef __cplusplus
    #define __PYX_EXTERN_C extern "C"
  #else
    #define __PYX_EXTERN_C extern
  #endif
#endif

__PYX_EXTERN_C DL_IMPORT(PyObject) *test(void);

#endif /* !__PYX_HAVE_API__RSTPython */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC initRSTPython(void);
#else
PyMODINIT_FUNC PyInit_RSTPython(void);
#endif

#endif /* !__PYX_HAVE__RSTPython */
