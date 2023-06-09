#include "xf_resize_config.h"

//static constexpr int __XF_DEPTH = (HEIGHT * WIDTH * (XF_PIXELWIDTH(TYPE, NPC_T)) / 8) / (INPUT_PTR_WIDTH / 8);
//static constexpr int __XF_DEPTH_OUT =
//    (NEWHEIGHT * NEWWIDTH * (XF_PIXELWIDTH(TYPE, NPC_T)) / 8) / (OUTPUT_PTR_WIDTH / 8);

void resize_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                  int rows_in,
                  int cols_in,
                  int rows_out,
                  int cols_out) {
// clang-format off
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=rows_in
    #pragma HLS INTERFACE s_axilite port=cols_in
    #pragma HLS INTERFACE s_axilite port=rows_out
    #pragma HLS INTERFACE s_axilite port=cols_out
    #pragma HLS INTERFACE s_axilite port=return
    // clang-format on

    xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC_T> in_mat(rows_in, cols_in);

    xf::cv::Mat<TYPE, NEWHEIGHT, NEWWIDTH, NPC_T> out_mat(rows_out, cols_out);

// clang-format off
    #pragma HLS DATAFLOW
    // clang-format on

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, TYPE, HEIGHT, WIDTH, NPC_T>(img_inp, in_mat);
    xf::cv::resize<INTERPOLATION, TYPE, HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH, NPC_T, MAXDOWNSCALE>(in_mat, out_mat);
    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, TYPE, NEWHEIGHT, NEWWIDTH, NPC_T>(out_mat, img_out);
}