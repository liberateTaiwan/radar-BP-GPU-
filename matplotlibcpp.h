#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244 4267)
#endif

namespace matplotlibcpp {
namespace detail {

// 删除所有现有的 select_npy_type 定义
#undef NPY_INT64
#undef NPY_UINT64

// 重新定义 select_npy_type
template <typename T> struct select_npy_type;
template <> struct select_npy_type<double> { static const NPY_TYPES type = NPY_DOUBLE; };
template <> struct select_npy_type<float> { static const NPY_TYPES type = NPY_FLOAT; };
template <> struct select_npy_type<bool> { static const NPY_TYPES type = NPY_BOOL; };
template <> struct select_npy_type<int8_t> { static const NPY_TYPES type = NPY_INT8; };
template <> struct select_npy_type<int16_t> { static const NPY_TYPES type = NPY_INT16; };
template <> struct select_npy_type<int32_t> { static const NPY_TYPES type = NPY_INT32; };
template <> struct select_npy_type<uint8_t> { static const NPY_TYPES type = NPY_UINT8; };
template <> struct select_npy_type<uint16_t> { static const NPY_TYPES type = NPY_UINT16; };
template <> struct select_npy_type<uint32_t> { static const NPY_TYPES type = NPY_UINT32; };

} // namespace detail
} // namespace matplotlibcpp

#ifdef _MSC_VER
#pragma warning(pop)
#endif
