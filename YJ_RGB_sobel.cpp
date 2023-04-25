//cmd:./dpu_main_code yolov5s_191.xmodel video_data_clean_new/speed_A/image_sequence/%03d.jpg -n1.5 -t 2
//cmd:./dpu_main_code yolov5s_YJ_rgb.xmodel video_data_clean_new/speed_A/val_12_month_rgb/images/%04d.jpg -n1.5 -t 2

/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <opencv2/imgproc/types_c.h>
#include <signal.h>
#include <sys/stat.h>
#include <cmath>
#include <iostream>
#include <ostream>
#include <cassert>
#include <chrono>
#include <future>
#include <memory>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <numeric>

#include <sstream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/time_measure.hpp>
#include <vitis/ai/stat_samples.hpp>
#include <vitis/ai/pointpillars.hpp>
#include <vitis/ai/bounded_queue.hpp>
#include <time.h>
#include "utils.cpp"
#include "xcl2.cpp"
#include "yolov3.cpp"
//#include "test_preprocess.h"

// RPU --->
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/mman.h>
#include <sys/timerfd.h>
#include <sys/types.h>
#define SHARED_RAM_POS_R0toR1 0x70000000
#define SHARED_RAM_POS_R1toR0 0x70100000
#define SHARED_RAM_POS_AtoR0 0x70200000
#define SHARED_RAM_POS_R0toA 0x70300000
#define SHARED_RAM_POS_AtoR1 0x70400000
#define SHARED_RAM_POS_R1toA 0x70500000
#define SHARED_RAM_LENGTH 0x100000
#define handle_error(msg) do { perror(msg); exit(-1); } while (0)
unsigned int Data_Test[4]={1,2,3,4};
// RPU <---

DEF_ENV_PARAM(DEBUG_DEMO, "0")
DEF_ENV_PARAM(DEMO_USE_VIDEO_WRITER, "0")
DEF_ENV_PARAM_2(
    DEMO_VIDEO_WRITER,
    "appsrc ! videoconvert ! queue ! kmssink "
    "driver-name=xlnx plane-id=39 fullscreen-overlay=false sync=false",
    std::string)

DEF_ENV_PARAM(DEMO_VIDEO_WRITER_WIDTH, "640")
DEF_ENV_PARAM(DEMO_VIDEO_WRITER_HEIGHT, "480")


DEF_ENV_PARAM(SAMPLES_ENABLE_BATCH, "1");
DEF_ENV_PARAM(SAMPLES_BATCH_NUM, "0");



using namespace std;
using namespace cv;

ofstream ofs;
mutex mtx;
bool index_record = false;

float FrmResize = 1;
class PPHandle {                                        //~One element per device
public:
  cl::Context contxt;
  cl::Device device;
  cl::Kernel kernel;
  cl::CommandQueue q;
  cl::Buffer paramsbuf;
};



unique_ptr<PPHandle> pp_kernel_init(                    //~初始化pipeline kernel
					char *xclbin,                                 //~參數：創建FPGA二進製文件（.xclbin）,構建FPGA二進制文件的關鍵是確定要生成的構建目標。
					const char *kernelName,                       //~參數：kernel可以用C / C ++或OpenCL C代碼描述，也可以從打包的RTL設計中創建。如上圖所示，每個硬件內核都獨立編譯為Xilinx目標（.xo）文件。
					int deviceIdx){                               //~參數：為讓Xilinx（.xo）文件與硬體平台鏈接，創建FPGA二進製文件（.xclbin），該文件會加載到目標平台上的Xilinx設備中，因此需要設備的索引值(deviceIdx)。
	    cout << "hello world!" << endl;

    // ------------------------------------------------------------------------------------
    // Step 1: Initialize the OpenCL environment
    // ------------------------------------------------------------------------------------
//	PPHandle *my_handle = new PPHandle;
//	handle = my_handle = (PPHandle *)my_handle;
    auto my_handle = make_unique<PPHandle>();           //~使用make_unique函式,來建構unique_ptr,nique_ptr 是一個藉由「移動move()」會將擁有權轉移到新的 unique_ptr
    cl_int err;                                         //~初始化openCL API的偵測錯誤值.(需要先取得 platform 的數目)因為要先取得系統上所有的 OpenCL platform. 在其它系統上，可能會有不同廠商提供的多個不同的 OpenCL platform => 因此需要先取得 platform 的數目
//  std::string binaryFile = (argc != 2) ? "dpu.xclbin" : argv[1];
    std::string binaryFile = xclbin;                    //~創建 FPGA二進製文件（.xclbin），該文件會加載到目標平台上的Xilinx設備中
    unsigned fileBufSize;

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the Xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices(); //~目標平台上的Xilinx設備
    devices.resize(1);                                  //~.resize(行數,列數),表示不改變列數，逗號都省略，只改變行數
    cl::Device device = devices[0];

    cl::Context context(device, NULL, NULL, NULL, &err);
    char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    cl::Program program(context, devices, bins, NULL, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl::Kernel krnl_xf_pp_pipeline_accel(program, "pp_pipeline_accel", &err); //~透過pipeline達成kernel功能之加速

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ------------------------------------------------------------------------------------
    // Create the buffers and allocate memory
//    cl::Buffer in1_buf(context, CL_MEM_ALLOC_HOST_PTR, sizeof(int) * DATA_SIZE, NULL, &err);
//    cl::Buffer in2_buf(context, CL_MEM_ALLOC_HOST_PTR, sizeof(int) * DATA_SIZE, NULL, &err);
//    cl::Buffer out_buf(context, CL_MEM_ALLOC_HOST_PTR, sizeof(int) * DATA_SIZE, NULL, &err);
//    cl::Buffer paramsbuf(context, CL_MEM_READ_ONLY , 9*4, NULL, &err);

    // Map buffers to kernel arguments, thereby assigning them to specific device memory banks
//    krnl_vector_add.setArg(0, in1_buf);
//    krnl_vector_add.setArg(1, in2_buf);
//    krnl_vector_add.setArg(2, out_buf);

  float params[9];
	params[0] = params[1] = params[2] = 0.0f;
//	params[3] = params[4] = params[5] = 0.00390625f;
	params[3] = params[4] = params[5] = 1.0f;
	params[6] = params[7] = params[8] = 0.0f;
                                                          //~透過openCL分配kernel參數們到指定的記憶體裝置上
	cl::Buffer paramsbuf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY , 9*4, params, &err);

  int th1=255,th2=255;
  krnl_xf_pp_pipeline_accel.setArg(8, paramsbuf);         //~透過pipeline達成kernel功能之加速功效
	krnl_xf_pp_pipeline_accel.setArg(9, th1);
	krnl_xf_pp_pipeline_accel.setArg(10, th2);
//	q.enqueueWriteBuffer(  paramsbuf,
//                         CL_TRUE,
//                         0,
//                         9*4,
//                         params);

  q.enqueueMigrateMemObjects({paramsbuf}, 0);            //~將buffer中的參數 分別排進佇列之中

	cout << "Initial Compelete!" << endl;
	my_handle->kernel = krnl_xf_pp_pipeline_accel;
	my_handle->contxt = context;
	my_handle->device = device;
	my_handle->q = q;
	my_handle->paramsbuf=paramsbuf;
	return std::move(my_handle);
}

unique_ptr<PPHandle> preprocess(unique_ptr<PPHandle> handle, cv::Mat img, int out_ht, int out_wt, cv::Mat image){
    struct timeval start_,end_, customization_start_, customization_end_;
	double lat_ = 0.0f, customization_lat_= 0.0f;
    gettimeofday(&start_, 0);
	int in_width,in_height;
	int out_width,out_height;
	in_width = img.cols;
	in_height = img.rows;
//	float* data =
//      new float[in_height*in_width*3];
//    std::vector<float[in_height*in_width*3]> data;
//	float *data_ptr = data;
	//output image dimensions 224x224
	out_height = out_ht;
	out_width = out_wt;
//

	float scale_height = (float)out_height/(float)in_height;
    float scale_width = (float)out_width/(float)in_width;
	int out_height_resize, out_width_resize;
    if(scale_width<scale_height){
    	out_width_resize = out_width;
    	out_height_resize = (int)((float)(in_height*out_width)/(float)in_width);
    }
    else
    {
    	out_width_resize = (int)((float)(in_width*out_height)/(float)in_height);
    	out_height_resize = out_height;
    }

//    int dx = (out_width - out_width_resize)/2;
//    int dy = (out_height - out_height_resize)/2;

    cl::Context context = handle->contxt;
	cl::Kernel krnl = handle->kernel;
	cl::Device device = handle->device;
	cl::CommandQueue q;//(context, device,CL_QUEUE_PROFILING_ENABLE);
	q = handle->q;
	cl::Buffer paramsbuf;//(context, CL_MEM_READ_ONLY,9*4);
	paramsbuf       = handle->paramsbuf;

    cl::Buffer imageToDevice(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, in_height*in_width*3, img.data);
	cl::Buffer imageFromDevice(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, out_height*out_width*3, image.data);

	krnl.setArg(0, imageToDevice);
	krnl.setArg(1, imageFromDevice);
	krnl.setArg(2, in_height);
	krnl.setArg(3, in_width);
	krnl.setArg(4, out_height_resize);
	krnl.setArg(5, out_width_resize);
	krnl.setArg(6, out_height);
	krnl.setArg(7, out_width);

//  Copy data from host to FPGA
//	q.enqueueWriteBuffer(
//                         imageToDevice,
//                         CL_TRUE,
//                         0,
//                         in_height*in_width*3,
//                         img.data);
//  gettimeofday(&customization_start_, 0);
    q.enqueueMigrateMemObjects({imageToDevice}, 0);


    // Profiling Objects
//cl_ulong start= 0;
//cl_ulong end = 0;
//	double diff_prof = 0.0f;
    cl::Event event_sp;

    q.enqueueTask(krnl, NULL,&event_sp);
	clWaitForEvents(1, (const cl_event*) &event_sp);
//	q.enqueueReadBuffer(
//                         imageFromDevice,
//                         CL_TRUE,
//                         0,
//                         out_height*out_width*3,
//                         image.data);

    q.enqueueMigrateMemObjects({imageFromDevice}, CL_MIGRATE_MEM_OBJECT_HOST);
//    gettimeofday(&customization_end_, 0);

    //Profiling
//	event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
//	event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
//	diff_prof = end-start;
//	std::cout<<"kernel latency = "<<(diff_prof/1000000)<<"ms"<<std::endl;
    q.finish();
//    for(int i=0;i<(3*out_width*out_height);i++)
//		data_ptr[i]=data_ptr[i]/256;
//    result = Mat(out_height, out_width, CV_32FC3, data);
//    imwrite("test_out.jpg", result);
//    delete[] data;
//    customization_lat_ = (customization_end_.tv_sec * 1e6 + customization_end_.tv_usec) - (customization_start_.tv_sec * 1e6 + customization_start_.tv_usec);
//    std::cout << "\n\n Custom latency " << customization_lat_ / 1000 << "ms" << std::endl;
//    gettimeofday(&end_, 0);
//    lat_ = (end_.tv_sec * 1e6 + end_.tv_usec) - (start_.tv_sec * 1e6 + start_.tv_usec);
//    std::cout << "\n\n Overall latency " << lat_ / 1000 << "ms" << std::endl;

    cout << "Preprocess Compelete!" << endl;
    return std::move(handle);
}

struct BenchMarkResult {
  long ret;
  vitis::ai::StatSamples e2eSamples;
  vitis::ai::StatSamples pp_e2e_stat_samples;
};

std::mutex g_mtx;
int g_num_of_threads = 1;
int g_num_of_seconds = 10;
std::string file_name = "";


std::string g_list_name = "ppbin.list";
std::string g_report_file_name = "";
long g_total = 0;
double g_e2e_mean = 0.0;
double g_pp_e2e_mean = 0.0;
bool g_stop = false;
std::atomic<int> _counter(0);
long act_time = 30000000;

const string yolov3_config = {
    "   name: \"yolov5s_191\" \n"
    "   model_type : YOLOv3 \n"
    "   yolo_v3_param { \n"
    "     num_classes: 3 \n"
    "     anchorCnt: 3 \n"
    "     conf_threshold: 0.65 \n"                       //信心程度:0.65 (try:0.01<x<0.08
    "     nms_threshold: 0.45 \n"                        //兩個框如果框太近(IoU值太大)，那就要刪掉一個框。
    "     layer_name: \"10812\" \n"
    "     layer_name: \"10765\" \n"
    "     layer_name: \"10718\" \n"
    "     biases: 10 \n"
    "     biases: 13 \n"
    "     biases: 16 \n"
    "     biases: 30 \n"
    "     biases: 33 \n"
    "     biases: 23 \n"
    "     biases: 30 \n"
    "     biases: 61 \n"
    "     biases: 62 \n"
    "     biases: 45 \n"
    "     biases: 59 \n"
    "     biases: 119 \n"
    "     biases: 116 \n"
    "     biases: 90 \n"
    "     biases: 156 \n"
    "     biases: 198 \n"
    "     biases: 373 \n"
    "     biases: 326 \n"
    "     test_mAP: true \n"
    "   } \n"};
static void signal_handler(int signal) { g_stop = true; }
static void usage() {                                   //~將命令列引數印出,引數依序為:
  std::cout << "usage: env dpbenchmark \n"
               " -l <log_file_name> \n"                 //~所紀錄的檔名
               " -t <num_of_threads> \n"                //~執行緒數量
               " -s <num_of_seconds> \n"                //~秒數
               " <points list file> \n"                 //~檔案的指標＝檔案所存放的位址
            << std::endl;                               //~換行和將緩衝區中的內容刷新到屏幕
}
inline void parse_opt(int argc, char *argv[]) {         //~[自訂函式]解析命令列引數(argc, argv):由字符串(optstring) 劃分命令列引數(argv)
  int opt = 0;                                          //~opt表示為選項(option)

  while ((opt = getopt(argc, argv, "t:s:l:n:r:")) != -1) {  //~函式getopt()用來分析命令列引數,不斷讀取命令列引數,直到讀取完為止
    switch (opt) {
      case 't':
        g_num_of_threads = std::stoi(optarg);           //~執行緒數量 = 數字字符串(string)轉換成整數(int)輸出
        break;
      case 's':
        g_num_of_seconds = std::stoi(optarg);           //~秒數  //extern char *optarg = 選項的引數指標 (getopt內部固定用法)
        break;
      case 'l':
        g_report_file_name = optarg;                    //~紀錄檔名
        break;
      case 'n':
        FrmResize = std::stof(optarg);                  //~縮放播放畫面之比例
        break;
      case 'r':
    	  index_record=true;
    	  file_name=optarg;
       break;
      default:
        usage();                                        //~自訂函數usage():將命令列引數印出
        exit(1);                                        //~退出程式exit是在呼叫處強行退出程式，執行一次程式就結束
    }
  }
  if (optind >= argc) {                                 //~optind = 初始化值為「1」，下一次呼叫getopt時，從optind儲存的位置重新開始檢查選項，也就是從下一個'-'的選項開始
    std::cerr << "Expected argument after options\n";   //~接續上行，若下一個開始選項，等於或超過命令列整數引數數量，則印出程式錯誤提示
    exit(EXIT_FAILURE);                                 //~EXIT_FAILURE = 1,表示異常退出，在退出前可以給出一些提示信息，或在調試程序中察看出錯原因
  }

  cout << "g_num_of_threads:" << g_num_of_threads << endl; //~印出執行續數量
  g_list_name = argv[argc - 1];                         //~將命令列字元引數，另存為新變數g_list_name
  return;                                               //~返回
}
int total_step = 0;
int step = 1;

static void report() {
  float sec = (float)act_time / 1000000.0;
  float fps = ((float)g_total) / sec;
  cout << "FPS=" << fps << "\n";
  cout << "ACT_TIME=" << act_time << "\n";
  cout << "E2E_MEAN=" << g_e2e_mean << "\n";
  cout << "PP_E2E_MEAN=" << g_pp_e2e_mean << "\n";
  return;
}
static void report_step() {
  float fps = ((float)total_step) / ((float)step);
  cout << "step " << step << "FPS=" << fps << "\n";
//  cout << std::flush;
  return;
}


struct FrameInfo {
  int channel_id;
  unsigned long frame_id;
  cv::Mat mat;            //~!!!
  cv::Mat preprocessed;            //~!!!
  float max_fps;            //~!!!
  float fps;            //~!!!
  int belonging;
  int mosaik_width;
  int mosaik_height;
  int horizontal_num;
  int vertical_num;
  cv::Rect_<int> local_rect;
  cv::Rect_<int> page_layout;
  std::string channel_name;
  vector<vector<float>> boxes;
};

using queue_t = vitis::ai::BoundedQueue<FrameInfo>;

///**********************************************************************************

struct MyThread {
  // static std::vector<MyThread *> all_threads_;
  static inline std::vector<MyThread*>& all_threads() {
    static std::vector<MyThread*> threads;
    return threads;
  };
  static void signal_handler(int) { stop_all(); }

  static void stop_all() {
    for (auto& th : all_threads()) {
      th->stop();
    }
  }
  static void wait_all() {
    for (auto& th : all_threads()) {
      th->wait();
    }
  }
  static void start_all() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "Thread num " << all_threads().size();
    for (auto& th : all_threads()) {
      th->start();
    }
  }

  static void main_proxy(MyThread* me) { return me->main(); }

  void main() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is started";
    while (!stop_) {
      auto run_ret = run();
      if (!stop_) {
        stop_ = run_ret != 0;
      }
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "thread [" << name() << "] is ended";
  }

  virtual int run() = 0;

  virtual std::string name() = 0;

  explicit MyThread() : stop_(false), thread_{nullptr} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT A Thread";
    all_threads().push_back(this);
  }

  virtual ~MyThread() {
    all_threads().erase(
        std::remove(all_threads().begin(), all_threads().end(), this),
        all_threads().end());
  }

  void start() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is starting";
    thread_ = std::unique_ptr<std::thread>(new std::thread(main_proxy, this));
  }

  void stop() {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] is stopped.";
    stop_ = true;
  }

  void wait() {
    if (thread_ && thread_->joinable()) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "waiting for [" << name() << "] ended";
      thread_->join();
    }
  }
  bool is_stopped() { return stop_; }

  bool stop_;
  std::unique_ptr<std::thread> thread_;
};


///**********************************************************************************


//Decoded thread
struct DecodeThread : public MyThread {
  DecodeThread(int channel_id, const std::string& video_file, queue_t* queue,
                unique_ptr<PPHandle> handle, int width, int height)
      : MyThread{},
        channel_id_{channel_id},
        video_file_{video_file},
        frame_id_{0},
        video_stream_{},
        queue_{queue},
        width_{width},
        height_{height}
  {
    handle_ = std::move(handle);
    open_stream();
    auto& cap = *video_stream_.get();
    if (is_camera_) {
      cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      cap.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
    }
  }

//  virtual ~DecodeThread() {}

  virtual int run() override {
    auto& cap = *video_stream_.get();
    cv::Mat grayscale;
    cap >> grayscale;

    auto video_ended = grayscale.empty();
    if (video_ended) {
      // loop the video
      open_stream();
      return 0;
    }
//    imwrite(std::string("RealShot_IR/images_choice_image_sequence/") + std::to_string(frame_id_) +  "_input.jpg", image);
    cv::Mat image;
    //**********************************************************
    //cvtColor(grayscale, image, CV_GRAY2BGR);
    image = grayscale;
    //**********************************************************
    cv::Mat preprocessed = Mat(height_, width_, CV_8UC3);
    handle_ = preprocess(std::move(handle_), image, height_, width_, preprocessed);
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "decode queue size " << queue_->size();
    if (queue_->size() > 0 && is_camera_ == true) {
      return 0;
    }
    while (!queue_->push(FrameInfo{channel_id_, ++frame_id_, image, preprocessed},
                         std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override {
    return std::string{"DedodeThread-"} + std::to_string(channel_id_);
  }

  void open_stream() {
    is_camera_ = video_file_.size() == 1 && video_file_[0] >= '0' &&
                 video_file_[0] <= '9';
    video_stream_ = std::unique_ptr<cv::VideoCapture>(
        is_camera_ ? new cv::VideoCapture(std::stoi(video_file_))
                   : new cv::VideoCapture(video_file_, 0));
    if (!video_stream_->isOpened()) {
      LOG(ERROR) << "cannot open file " << video_file_;
      stop();
    }
    cout << "video_file_ : " << video_file_ << endl;
  }

  int channel_id_;
  std::string video_file_;
  unsigned long frame_id_;
  std::unique_ptr<cv::VideoCapture> video_stream_;         //~OpenCV 擷取網串流影像,串流影像video_stream_
  queue_t* queue_;
  bool is_camera_;
  unique_ptr<PPHandle> handle_;
  int height_;
  int width_;
};


///**********************************************************************************

//dpu thread
template <typename dpu_model_type>
struct DpuFilter{
  DpuFilter(std::unique_ptr<dpu_model_type> &&dpu_model):
  dpu_model_{std::move(dpu_model)} {}
  virtual ~DpuFilter() {}
  std::unique_ptr<dpu_model_type> dpu_model_;
};



Mat sobel_func(Mat target_part){
    // read image
    Mat src = target_part;
    if (src.empty()){
        cout << "could not load image." << endl;
        exit(0);
    }
    // namedWindow("src", WINDOW_AUTOSIZE);
    // imshow("src", src);

    // 1. gaussian blur
    Mat srcBlur;
    GaussianBlur(src, srcBlur, Size(3, 3) , 0, 0);

    // 2. bgr to gray
    Mat srcGray;
    cvtColor(src, srcGray, COLOR_BGR2GRAY);

    // 3. get x, y grad
    Mat gradX, gradY;
    Sobel(srcGray, gradX, CV_16S, 1, 0, 3);
    Sobel(srcGray, gradY, CV_16S, 0, 1, 3);
    // Scharr(srcGray, gradX, CV_16S, 1, 0);
    // Scharr(srcGray, gradY, CV_16S, 0, 1);
    convertScaleAbs(gradX, gradX);  // calculates absolute values, and converts the result to 8-bit.
    convertScaleAbs(gradY, gradY);
    // namedWindow("gradY", WINDOW_AUTOSIZE);
    // imshow("gradX", gradX);
    // namedWindow("gradY", WINDOW_AUTOSIZE);
    // imshow("gradY", gradY);

    // printf("type: %d, %d", gradX.type(), gradY.type());

    // 4. gradx grady fuse
    Mat dst;
    addWeighted(gradX, 0.5, gradY, 0.5, 0, dst);
    // namedWindow("dst", WINDOW_AUTOSIZE);
    // imshow("dst", dst);

    // 4.1
    Mat gradXY = Mat(gradX.size(), gradX.type());
    for (int row = 0; row < gradX.rows; row++){
        for (int col = 0; col < gradX.cols; col++){
            int gX = gradX.at<uchar>(row, col);
            int gY = gradY.at<uchar>(row, col);
            gradXY.at<uchar>(row, col) = saturate_cast<uchar>(gX + gY);
        }
    }

    threshold(gradXY, gradXY, 0, 255, THRESH_OTSU);

    // erode(gradXY, gradXY, Mat(), Point(-1, -1), 1);
    // dilate(gradXY, gradXY, Mat(), Point(-1, -1), 5);

    Mat blank_ch, fin_img;
    blank_ch = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
    std::vector<cv::Mat> channels_g;
    channels_g.push_back(blank_ch);
    channels_g.push_back(gradXY);
    channels_g.push_back(blank_ch);
    cv::merge(channels_g, fin_img);


    Mat mix = Mat(src.size(), src.type());
    // Mat gradXYrgb = Mat(src.size(), src.type());
    // cvtColor(gradXY, gradXYrgb, COLOR_GRAY2BGR);
    bitwise_or(src, fin_img, mix);

    //resize(mix,mix,Size(mix.cols*5,mix.rows*5),0,0,INTER_LINEAR);
    //resize(fin_img,fin_img,Size(fin_img.cols*5,fin_img.rows*5),0,0,INTER_LINEAR);

    // namedWindow("mix", WINDOW_AUTOSIZE);
    // imshow("mix", mix);
    // imshow("fin_img", fin_img);
    // imwrite("result.jpg", mix);
    // imwrite("gradXY.jpg", gradXY);
    // waitKey(0);
    return mix;
}

///**********************************************************************************
//root@zcu104_custom_plnx_addConfig:~/YJ_RGB_demo# ./demo.sh
//./dpu_main_code yolov5s_191.xmodel video_data_clean_new/speed_A/val_12_month_rgb/images/%04d.jpg -n1.5

//DpuThread thread
template <typename dpu_model_type>
struct DpuThread : public MyThread {
  DpuThread(std::unique_ptr<dpu_model_type>&& model, queue_t* queue_in,
            queue_t* queue_out, const std::string& suffix)
      : MyThread{},
        model_{std::move(model)},
        queue_in_{queue_in},
        queue_out_{queue_out},
        suffix_{suffix} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT DPU";
  }

  virtual ~DpuThread() {} //~***no reason :)



  virtual int run() override {
	mtx.lock();

    FrameInfo frame;
    if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
      return 0;
    }





    //原圖:等比例放大
    resize(frame.mat, frame.mat, Size(0, 0), FrmResize, FrmResize);
    FrameInfo frame_info;
//    cout<<frame_info.frame_id<<"frame_id ~~~~~~"<<endl;




//    ofs<<"ddd" <<"\n";




    if (model_) {
      // RPU --->
      int dh = open("/dev/mem", O_RDWR | O_SYNC); // Open /dev/mem which represents the whole physical memory
      unsigned int* RtnR0 = (unsigned int *)mmap(NULL, SHARED_RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, dh, SHARED_RAM_POS_R0toA);
      unsigned int* RtnR1 = (unsigned int *)mmap(NULL, SHARED_RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, dh, SHARED_RAM_POS_R1toA);
      unsigned int* ToR0 = (unsigned int *)mmap(NULL, SHARED_RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, dh, SHARED_RAM_POS_AtoR0);
      unsigned int* ToR1 = (unsigned int *)mmap(NULL, SHARED_RAM_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, dh, SHARED_RAM_POS_AtoR1);
      unsigned int tmpCnt;




      // RPU <---

      // model initialize + detection
      // Set the mean values and scale values.
      model_->setMeanScaleBGR({0.0f, 0.0f, 0.0f},
                        {0.00390625f, 0.00390625f, 0.00390625f});
      auto input_tensor = model_->getInputTensor(0u);
      CHECK_EQ((int)input_tensor.size(), 1)
        << "the dpu model must have only one input";
      int width = input_tensor[0].width;
      int height = input_tensor[0].height;
      // Create a config and set the correlating data to control post-process.
      vitis::ai::proto::DpuModelParam config;
      // Fill all the parameters.
      auto ok = google::protobuf::TextFormat::ParseFromString(yolov3_config, &config);
      cout << frame.preprocessed.type() << endl;
      // 圖像像素填充
      model_->setImageRGB(frame.preprocessed);
      // inference
      model_->run(0u);
      // postprocessing
      auto output_tensor = model_->getOutputTensor(0u);
      auto results = vitis::ai::yolov3_post_process(
        input_tensor, output_tensor, config, frame.mat.cols, frame.mat.rows);

      // draw box
      // cv::Mat original_img;
      // frame.mat.copyTo(original_img);
      std::vector<decltype(Scalar(0, 255, 0))> label_color;
      label_color.emplace_back(Scalar(0, 0, 255));
      label_color.emplace_back(Scalar(255, 0, 0));
      label_color.emplace_back(Scalar(0, 255, 0));

      std::vector<std::string> label_text;
      label_text.emplace_back("target");
      label_text.emplace_back("small car");
      label_text.emplace_back("large car");
      cout<<frame.frame_id<<"frame_id ~~~~~~"<<endl;


      if(index_record){
              string strr;
              stringstream ss;
              ss << setw(6) << setfill('0') << frame.frame_id ;
              ss>>strr;
              ofs.open("ai_result/labels/"+strr+".txt",ios::trunc);
              }


      for(auto& box : results.bboxes) {
        int label = box.label;
        float xmin = box.x * frame.mat.cols + 1;//box:原圖0-1%
        float ymin = box.y * frame.mat.rows + 1;
        float xmax = xmin + box.width * frame.mat.cols;
        float ymax = ymin + box.height * frame.mat.rows;
        float xctrl=box.x+box.width/2;
        float yctrl=box.y+box.height/2;

//		cout<<frame.mat.rows<<"~~~~row"<<endl;
//        cout<<box.x<<"~~~~x"<<endl;
//        cout<<box.y<<"~~~~y"<<endl;
//        cout<<box.x+box.width/2<<"~~~~x central"<<endl;
//        cout<<box.y+box.height/2<<"~~~~y central"<<endl;
//
//        cout<<box.width<<"~~~~width"<<endl;
//        cout<<box.height<<"~~~~height"<<endl;

        if (xmin < 0.) xmin = 1.;
        if (ymin < 0.) ymin = 1.;
        if (xmax > frame.mat.cols) xmax = frame.mat.cols;
        if (ymax > frame.mat.rows) ymax = frame.mat.rows;
        float confidence = box.score;


        cout <<"Result:"<< label << "\t" << xmin << "\t" << ymin << "\t"
              << xmax << "\t" << ymax << "\t" << confidence << "\n";



        int line_length = 0;
        int lx = xmax - xmin;
        int ly = ymax - ymin;
        if (lx > ly) line_length = ly;
        else line_length = lx;

        // RPU --->
        // casting
        unsigned int uns_xmin = (unsigned int) xmin;
        unsigned int uns_ymin = (unsigned int) ymin;
        unsigned int uns_xmax = (unsigned int) xmax;
        unsigned int uns_ymax = (unsigned int) ymax;



//        unsigned int uns_xctrl = (unsigned int) xctrl;
//        unsigned int uns_yctrl = (unsigned int) yctrl;
//        unsigned int uns_width = (unsigned int) box.width;
//        unsigned int uns_height = (unsigned int) box.height;
//        unsigned int uns_oriconfidence = (unsigned int) confidence;

//        confidence = confidence*100;
        unsigned int uns_confidence = (unsigned int) confidence*100;
        unsigned int uns_label = (unsigned int) label;
        // 位移
        ToR0[0] = uns_xmin;
        ToR0[1] = uns_ymin;
        ToR0[2] = uns_xmax;
        ToR0[3] = uns_ymax;
        ToR0[4] = uns_confidence;
        ToR0[5] = uns_label;
        cout << "CHECK: uns_label | uns_xmin | uns_ymin | uns_xmax  | uns_ymax | uns_confidence | tmpCnt | ToR0[0] | ToR1[0]\n";
        cout << "CHECK: " << uns_label << "\t" << uns_xmin << "\t" << uns_ymin << "\t"
              << uns_xmax << "\t" << uns_ymax << "\t" << uns_confidence << "\t"<< tmpCnt << "\t"<< ToR0[0]<< "\t" << ToR1[0] << "\n";
        // RPU <---

        if(index_record){

        ofs << uns_label << " " << xctrl << " " << yctrl << " "
                        << box.width << " " << box.height << " " << confidence << "\n";
        }

        if(label == 0){    //label_text[label] = "target"
          //取得影像的某個部分
          auto croppedImage = frame.mat(Rect(xmin, ymin, lx, ly));

          //將sobel處理加入合併之小影像
          Mat sobel_result;
          sobel_result = sobel_func(croppedImage);

          //子圖:等比例放大
          int magnify = 4;
          Mat tmp_croppedImage;
          resize(sobel_result, tmp_croppedImage, Size(0, 0), magnify, magnify);

          //指定插入的大小和位置 依序是:左上、左下、右下
          auto backImage = frame.mat;
          //auto addImage = backImage(Rect(30,30,croppedImage.cols,croppedImage.rows));
          //auto addImage = backImage(Rect(1, (frame.mat.rows-tmp_croppedImage.rows-1), tmp_croppedImage.cols, tmp_croppedImage.rows));
          Mat addImage = backImage(Rect((frame.mat.cols-tmp_croppedImage.cols-1), (frame.mat.rows-tmp_croppedImage.rows-1), tmp_croppedImage.cols, tmp_croppedImage.rows));

          //合併大小不同的影像:類似把一個小Logo加到原本影像上，且能夠指定Logo的位置
          //addWeighted(addImage,0.5, croppedImage, 0.5, 0, addImage);
          addWeighted(addImage,0.01, tmp_croppedImage, 0.99, 0, addImage);

          //子圖加入外框框線
          rectangle(frame.mat, Point(frame.mat.cols-3, frame.mat.rows-2) - Point(tmp_croppedImage.cols, tmp_croppedImage.rows) , Point(frame.mat.cols-2, frame.mat.rows-2), Scalar(0,0,255), 2.95, 1, 0);

          // 1. 合併大小不同的影像: https://www.796t.com/content/1549893608.html
          // 2. 圖像的縮放: https://blog.csdn.net/i_chaoren/article/details/54564663.html
        }

          int draw_line_length = int(line_length*0.3);
          line(frame.mat, Point(xmin, ymin), Point(xmin, ymin+draw_line_length),
                    label_color[label], 1.5, cv::LINE_AA, 0);
          line(frame.mat, Point(xmin, ymin), Point(xmin+draw_line_length, ymin),
                    label_color[label], 1.5, cv::LINE_AA, 0);
          line(frame.mat, Point(xmax, ymin), Point(xmax, ymin+draw_line_length),
                    label_color[label], 1.5, cv::LINE_AA, 0);
          line(frame.mat, Point(xmax, ymin), Point(xmax-draw_line_length, ymin),
                    label_color[label], 1.5, cv::LINE_AA, 0);
          line(frame.mat, Point(xmin, ymax), Point(xmin+draw_line_length, ymax),
                    label_color[label], 1.5, cv::LINE_AA, 0);
          line(frame.mat, Point(xmin, ymax), Point(xmin, ymax-draw_line_length),
                    label_color[label], 1.5, cv::LINE_AA, 0);
          line(frame.mat, Point(xmax, ymax), Point(xmax-draw_line_length, ymax),
                    label_color[label], 1.5, cv::LINE_AA, 0);
          line(frame.mat, Point(xmax, ymax), Point(xmax, ymax-draw_line_length),
                    label_color[label], 1.5, cv::LINE_AA, 0);

          std::string show_text = label_text[label] + " " + std::to_string(confidence).substr(0, 4);
          int baseline=0;
          Size textSize = getTextSize(show_text, cv::FONT_HERSHEY_DUPLEX, 0.3, 1, &baseline);
          //rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
          rectangle(frame.mat, Point(xmin, ymin) - Point(0, textSize.height), Point(xmin, ymin) + Point(textSize.width, 0), label_color[label], cv::FILLED);
          putText(frame.mat, show_text, Point(xmin, ymin),   cv::FONT_HERSHEY_DUPLEX, 0.3, Scalar(225, 255, 255), 1, cv::LINE_AA, 0);

      }ofs.close();
      }



    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "dpu queue size " << queue_out_->size();
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
//    ofs.close();
    mtx.unlock();


    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  std::unique_ptr<dpu_model_type> model_;
  std::unique_ptr<dpu_model_type> task;
  queue_t* queue_in_;
  queue_t* queue_out_;
  std::string suffix_;
};

///**********************************************************************************

// sort thread
struct SortingThread : public MyThread {
  SortingThread(queue_t* queue_in, queue_t* queue_out,
                const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        queue_out_{queue_out},
        frame_id_{0},
        suffix_{suffix},
        fps_{0.0f},
        max_fps_{0.0f} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT SORTING";
  }
  virtual ~SortingThread() {}
  virtual int run() override {
    FrameInfo frame;
    frame_id_++;
    auto frame_id = frame_id_;
    auto cond =
        std::function<bool(const FrameInfo&)>{[frame_id](const FrameInfo& f) {
          // sorted by frame id
          return f.frame_id <= frame_id;
        }};
//        std::function<void(int, int)> callback_keyevent = nullptr;
    if (!queue_in_->pop(frame, cond, std::chrono::milliseconds(500))) {
      return 0;
    }
    auto now = std::chrono::steady_clock::now();
    float fps = -1.0f;
    long duration = 0;
    if (!points_.empty()) {
      auto end = points_.back();
      duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - end)
              .count();
      float duration2 = (float)duration;
      float total = (float)points_.size();
      fps = total / duration2 * 1000.0f;    //      calculate fps
      auto x = 10;
      auto y = 20;
      fps_ = fps;
      frame.fps = fps;
      max_fps_ = std::max(max_fps_, fps_);
      frame.max_fps = max_fps_;
      if (frame.mat.cols > 200)
        cv::putText(frame.mat, std::string("FPS: ") + std::to_string(fps),
                    cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(20, 20, 180), 2, 1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "thread [" << name() << "] "
        << " frame id " << frame.frame_id << " sorting queue size "
        << queue_out_->size() << "   FPS: " << fps;
    points_.push_front(now);
    if (duration > 2000) {  // sliding window for 2 seconds.
      points_.pop_back();
    }
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string{"SORT-"} + suffix_; }
  queue_t* queue_in_;
  queue_t* queue_out_;
  unsigned long frame_id_;
  std::deque<std::chrono::time_point<std::chrono::steady_clock>> points_;
  std::string suffix_;
  float fps_;
  float max_fps_;
};



///**********************************************************************************

static std::unique_ptr<cv::VideoWriter> maybe_create_gst_video_writer(
    int width, int height) {
  if (!ENV_PARAM(DEMO_USE_VIDEO_WRITER)) {
    return nullptr;
  }
  auto pipeline = ENV_PARAM(DEMO_VIDEO_WRITER);
  auto video_stream = std::unique_ptr<cv::VideoWriter>(new cv::VideoWriter(
      pipeline, cv::CAP_GSTREAMER, 0, 25.0, cv::Size(width, height), true));
  auto& writer = *video_stream.get();
  if (!writer.isOpened()) {
    LOG(FATAL) << "cannot open gst: " << pipeline;
    return nullptr;
  } else {
    LOG(INFO) << "video writer is created: " << width << "x" << height << " "
              << pipeline;
  }
  return video_stream;
}

///**********************************************************************************

//GUI thread
struct GuiThread : public MyThread {
  static std::shared_ptr<GuiThread> instance() {
    static std::weak_ptr<GuiThread> the_instance;
    std::shared_ptr<GuiThread> ret;
    if (the_instance.expired()) {
      ret = std::make_shared<GuiThread>();
      the_instance = ret;
    }
    ret = the_instance.lock();
    assert(ret != nullptr);
#if USE_DRM
//    vitis::ai::imshow_open();
    cout << "is using DRM" << endl;
    cv::Mat img = gui_background();
    imshow_set_background(img);
#endif
    return ret;
  }

  GuiThread()
      : MyThread{},
        queue_{
            new queue_t{
                10}  // assuming GUI is not bottleneck, 10 is high enough
        },
        inactive_counter_{0},
        video_writer_{maybe_create_gst_video_writer(
            ENV_PARAM(DEMO_VIDEO_WRITER_WIDTH),
            ENV_PARAM(DEMO_VIDEO_WRITER_HEIGHT))} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT GUI";
  }
  virtual ~GuiThread() {  //
#if USE_DRM
//    vitis::ai::imshow_close();
#endif
  }
  void clean_up_queue() {
    FrameInfo frame_info;
    while (!queue_->empty()) {
      queue_->pop(frame_info);
      frames_[frame_info.channel_id].frame_info = frame_info;
      frames_[frame_info.channel_id].dirty = true;
    }
  }
  virtual int run() override {
    FrameInfo frame_info;
    if (!queue_->pop(frame_info, std::chrono::milliseconds(500))) {
      inactive_counter_++;
      if (inactive_counter_ > 10) {
        // inactive for 5 second, stop
        LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "no frame_info to show";
        return 1;
      } else {
        return 0;
      }
    }

    inactive_counter_ = 0;
    frames_[frame_info.channel_id].frame_info = frame_info;
    frames_[frame_info.channel_id].dirty = true;
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << " gui queue size " << queue_->size()
        << ", state = " << (is_stopped() ? "stopped" : "running");
    clean_up_queue();
#if USE_DRM
    bool all_dirty = true;
    for (auto& f : frames_) {
      all_dirty = all_dirty && f.second.dirty;
    }
    if (!all_dirty) {
      // only show frames until all channels are dirty
      return 0;
    }
    auto width = modeset_get_fb_width();
    auto height = modeset_get_fb_height();
    auto screen_size = cv::Size{width, height};
    auto sizes = std::vector<cv::Size>(frames_.size());
    std::transform(frames_.begin(), frames_.end(), sizes.begin(),
                   [](const decltype(frames_)::value_type& a) {
                     return a.second.frame_info.mat.size();
                   });
    std::vector<cv::Rect> rects;
    rects = gui_layout();
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "rects size is  " << rects.size();

    for (const auto& rect : rects) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
          << "screen " << screen_size << "; r = " << rect;
      if ((rect.x + rect.width > width) || (rect.y + rect.height > height) ||
          (rect.x + rect.width < 1) || (rect.y + rect.height < 1)) {
        LOG(FATAL) << "out of boundary";
      }
    }
    int c = 0;
    for (auto& f : frames_) {
//      vitis::ai::imshow(rects[c], f.second.frame_info.mat);
      f.second.dirty = false;
      c++;
    }
//    vitis::ai::imshow_update();
#else
    bool any_dirty = false;
    for (auto& f : frames_) {
      if (f.second.dirty) {
        if (video_writer_ == nullptr) {
          cv::imshow(std::string{"CH-"} +
                         std::to_string(f.second.frame_info.channel_id),
                     f.second.frame_info.mat);
        } else {
          *video_writer_ << f.second.frame_info.mat;
        }
        f.second.dirty = false;
        any_dirty = true;
      }
    }
    if (video_writer_ == nullptr) {
      if (any_dirty) {
        auto key = cv::waitKey(1);
        if (key == 27) {
          return 1;
        }
      }
    }
#endif
    clean_up_queue();
    return 0;
  }

  virtual std::string name() override { return std::string{"GUIThread"}; }

  queue_t* getQueue() { return queue_.get(); }

  std::unique_ptr<queue_t> queue_;
  int inactive_counter_;
  struct FrameCache {
    bool dirty;
    FrameInfo frame_info;
  };
  std::map<int, FrameCache> frames_;
  std::unique_ptr<cv::VideoWriter> video_writer_;
};


///**********************************************************************************



//~cmd: input[1]=kernel_name, input[1]=kernel_name,
int main(int argc, char* argv[]) {

    auto kernel_name = argv[1];  //~讀取命令列引數argv[1],argv[1]=已訓練好的神經網路yolov5s_191.xmodel
    parse_opt(argc, argv);    //~[自訂函式]解析命令列引數(argc, argv):由字符串(optstring) 劃分命令列引數(argv)
    cout << "g_list_name:" << g_list_name << endl;        //~印出所紀錄的檔名等相關訊息
    LOG(INFO) << "log test" ;                             //~LOG()是一個記錄應用程式訊息的函式,記錄的訊息共四種： INFO: 0,  WARNING: 1,  ERROR: 2  FATAL: 3,可以將記錄信息在terminal顯示
    cout << "kernel loading" << endl;                     //~印出"正準備啟用kernel"
    auto task = vitis::ai::DpuTask::create(kernel_name);  //~啟用dpu kernel 或 讀取已訓練好的神經網路model,在此先預設為前者(?),並將此設為自動變數task(自動變數:自動推斷該變量的類型)
    cout << "kernel loaded!" << endl;                     //~印出"啟用kernel完畢"
    auto input_tensor = task->getInputTensor(0u);         //~將kernel所取得的輸入,設型態為自動的變數'輸入張量' , 0u???
    int width = input_tensor[0].width;                    //~取得第0個輸入張量的寬,設型態為整數的變數'寬'
    int height = input_tensor[0].height;                  //~取得第0個輸入張量的高,設型態為整數的變數'高'
//  PPHandle handle;                                      //~下行參數註解：char *xclbin,const char *kernelName,int deviceIdx
    auto handle = pp_kernel_init("/media/sd-mmcblk0p1/dpu.xclbin", "pp_pipeline_accel",0); //~[自訂函式]初始化kernel,並將此返回值,設型態為自動的變數'輸入張量'
    using model_t = typename decltype(task)::element_type; //~在此藉由decltype取得task型態,並將此取得的型態設為變數model_t,model_t代表意思是model的type


    auto channel_id = 0;                                  //~將型態為「自動變數」的channel_id,初始化 型態為「整數變數」
    auto decode_queue = std::unique_ptr<queue_t>{new queue_t{5}}; //~創建並初始化 decode_queue, decode_queue的型態為「queue_t」，並使用unique_ptr做不共享管理
    auto decode_thread = std::unique_ptr<DecodeThread>(   //~創建並初始化 decode_thread, decode_thread的型態為「DecodeThread」，，並在decode_thread中取得decode_queue的資訊，以及使用unique_ptr做不共享管理
        new DecodeThread{channel_id, g_list_name, decode_queue.get(), std::move(handle), width, height});

    auto dpu_thread = std::vector<std::unique_ptr<DpuThread<model_t>>>{}; //~創建並初始化 dpu_thread,dpu_thread是vector(向量),類型是<std::unique_ptr<DpuThread<model_t>>>,即為由unique_ptr管理的DpuThread類型,而此DpuThread類型需要參數model_t
    auto sorting_queue = std::unique_ptr<queue_t>(new queue_t(500 * g_num_of_threads));
                                                                          //~創建並初始化 sorting_queue, sorting_queue的型態為「queue_t」，並使用unique_ptr做不共享管理


    for (int i = 0; i < g_num_of_threads; ++i) {          //~使用多線程達到同步執行,使執行速度更快速
      dpu_thread.emplace_back(new DpuThread<model_t>(     //~在dpu_thread序列尾部創建一个元素
          std::unique_ptr<model_t>(vitis::ai::DpuTask::create(kernel_name)),
          decode_queue.get(), sorting_queue.get(), std::to_string(i)));
    }


    auto gui_thread = GuiThread::instance();
    auto gui_queue = gui_thread->getQueue();
    auto sorting_thread = std::unique_ptr<SortingThread>(
        new SortingThread(sorting_queue.get(), gui_queue, std::to_string(0)));

    MyThread::start_all();
    gui_thread->wait();
    MyThread::stop_all();
    MyThread::wait_all();

}
