#ifndef _XF_RESIZE_CONFIG_
#define _XF_RESIZE_CONFIG_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.hpp"
#include "xf_config_params.h"
#include "imgproc/xf_resize.hpp"

/* Interface types*/
#if RO

#if RGB
#define NPC_T XF_NPPC4
#else
#define NPC_T XF_NPPC8
#endif

#else
#define NPC_T XF_NPPC1
#endif

#if RGB
#define TYPE XF_8UC3
#define CH_TYPE XF_RGB
#else
#define TYPE XF_8UC1
#define CH_TYPE XF_GRAY
#endif

#define INPUT_PTR_WIDTH 128
#define OUTPUT_PTR_WIDTH 128

void resize_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                  int rows_in,
                  int cols_in,
                  int rows_out,
                  int cols_out);

#endif