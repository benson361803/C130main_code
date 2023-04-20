

DEF_ENV_PARAM(DEBUG_DEMO, "0")

int g_num_of_threads = 1;

using namespace std;
using namespace cv;

class PPHandle {
public:

  cl::Context contxt;
  cl::Device device;
  cl::Kernel kernel;
  cl::CommandQueue q;
  cl::Buffer paramsbuf;

};

const string yolov3_config = {
    "   name: \"yolov5s_191\" \n"
    "   model_type : YOLOv3 \n"
    "   yolo_v3_param { \n"
    "     num_classes: 3 \n"
    "     anchorCnt: 3 \n"
    "     conf_threshold: 0.3 \n"
    "     nms_threshold: 0.45 \n"
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

struct FrameInfo {
  int channel_id;
  unsigned long frame_id;
  cv::Mat mat;
  cv::Mat preprocessed;
  float max_fps;
  float fps;
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

  virtual ~MyThread() {  //
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

unique_ptr<PPHandle> pp_kernel_init(
					char *xclbin,
					const char *kernelName,
					int deviceIdx){
	    cout << "hello world!" << endl;

    // ------------------------------------------------------------------------------------
    // Step 1: Initialize the OpenCL environment
    // ------------------------------------------------------------------------------------
//	PPHandle *my_handle = new PPHandle;
//	handle = my_handle = (PPHandle *)my_handle;
    auto my_handle = make_unique<PPHandle>();
    cl_int err;
//    std::string binaryFile = (argc != 2) ? "dpu.xclbin" : argv[1];
    std::string binaryFile = xclbin;
    unsigned fileBufSize;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    devices.resize(1);
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    char *fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    cl::Program program(context, devices, bins, NULL, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl::Kernel krnl_xf_pp_pipeline_accel(program, "pp_pipeline_accel", &err);

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ----------------------- -------------------------------------------------------------
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

	cl::Buffer paramsbuf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY , 9*4, params, &err);

    int th1=255,th2=255;
    krnl_xf_pp_pipeline_accel.setArg(8, paramsbuf);
	krnl_xf_pp_pipeline_accel.setArg(9, th1);
	krnl_xf_pp_pipeline_accel.setArg(10, th2);
//	q.enqueueWriteBuffer(
//                         paramsbuf,
//                         CL_TRUE,
//                         0,
//                         9*4,
//                         params);

    q.enqueueMigrateMemObjects({paramsbuf}, 0);

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

	//Copy data from host to FPGA
//	q.enqueueWriteBuffer(
//                         imageToDevice,
//                         CL_TRUE,
//                         0,
//                         in_height*in_width*3,
//                         img.data);
    gettimeofday(&customization_start_, 0);
    q.enqueueMigrateMemObjects({imageToDevice}, 0);


    // Profiling Objects
	cl_ulong start= 0;
	cl_ulong end = 0;
	double diff_prof = 0.0f;
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
    gettimeofday(&customization_end_, 0);

    //Profiling
	event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
	event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
	diff_prof = end-start;
	std::cout<<"kernel latency = "<<(diff_prof/1000000)<<"ms"<<std::endl;
    q.finish();
//    for(int i=0;i<(3*out_width*out_height);i++)
//		data_ptr[i]=data_ptr[i]/256;
//    result = Mat(out_height, out_width, CV_32FC3, data);
//    imwrite("test_out.jpg", result);
//    delete[] data;
    customization_lat_ = (customization_end_.tv_sec * 1e6 + customization_end_.tv_usec) - (customization_start_.tv_sec * 1e6 + customization_start_.tv_usec);
    std::cout << "\n\n Custom latency " << customization_lat_ / 1000 << "ms" << std::endl;
    gettimeofday(&end_, 0);
    lat_ = (end_.tv_sec * 1e6 + end_.tv_usec) - (start_.tv_sec * 1e6 + start_.tv_usec);
    std::cout << "\n\n Overall latency " << lat_ / 1000 << "ms" << std::endl;

    cout << "Preprocess Compelete!" << endl;
    return std::move(handle);
}

//dpu thread
struct DpuThread : public MyThread {
  DpuThread(queue_t* queue_in, queue_t* queue_out, const std::string& suffix)
      : MyThread{},
        queue_in_{queue_in},
        queue_out_{queue_out},
        suffix_{suffix} {
    model_ = vitis::ai::DpuTask::create("yolov5s_191.xmodel");
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO)) << "INIT DPU";
  }
  virtual ~DpuThread() {}

  virtual int run() override {
    FrameInfo frame;
    if (!queue_in_->pop(frame, std::chrono::milliseconds(500))) {
      return 0;
    }

    FrameInfo frame_info;
    double customization_lat_= 0.0f;
    struct timeval customization_start_, customization_end_;

    if (model_) {
      // model initialize
        //detection
      model_->setMeanScaleBGR({0.0f, 0.0f, 0.0f},
                        {0.00390625f, 0.00390625f, 0.00390625f});
      auto input_tensor = model_->getInputTensor(0u);
      CHECK_EQ((int)input_tensor.size(), 1)
        << " the dpu model must have only one input";
      int width = input_tensor[0].width;
      int height = input_tensor[0].height;
      vitis::ai::proto::DpuModelParam config;
      auto ok = google::protobuf::TextFormat::ParseFromString(yolov3_config, &config);
      cout << frame.preprocessed.type() << endl;
      gettimeofday(&customization_start_, 0);
      model_->setImageRGB(frame.preprocessed);
      // inference
      model_->run(0u);
      // postprocessing
      auto output_tensor = model_->getOutputTensor(0u);
      gettimeofday(&customization_end_, 0);
      customization_lat_ = (customization_end_.tv_sec * 1e6 + customization_end_.tv_usec) - (customization_start_.tv_sec * 1e6 + customization_start_.tv_usec);
      std::cout << "\n\n Custom latency " << customization_lat_ / 1000 << "ms" << std::endl;
      auto results = vitis::ai::yolov3_post_process(
        input_tensor, output_tensor, config, frame.mat.cols, frame.mat.rows);
      // draw box
//      cv::Mat original_img;
//      frame.mat.copyTo(original_img);
       std::vector<decltype(Scalar(0, 255, 0))> label_color;
        label_color.emplace_back(Scalar(0, 0, 255));
        label_color.emplace_back(Scalar(255, 0, 0));
        label_color.emplace_back(Scalar(0, 255, 0));

        std::vector<std::string> label_text;
        label_text.emplace_back("target");
        label_text.emplace_back("small car");
        label_text.emplace_back("large car");
//        cvtColor(frame.mat, frame.mat, CV_RGB2BGR);
      for (auto& box : results.bboxes) {
            int label = box.label;
            float xmin = box.x * frame.mat.cols + 1;
            float ymin = box.y * frame.mat.rows + 1;
            float xmax = xmin + box.width * frame.mat.cols;
            float ymax = ymin + box.height * frame.mat.rows;
            if (xmin < 0.) xmin = 1.;
            if (ymin < 0.) ymin = 1.;
            if (xmax > frame.mat.cols) xmax = frame.mat.cols;
            if (ymax > frame.mat.rows) ymax = frame.mat.rows;
            float confidence = box.score;

            cout << "RESULT: " << label << "\t" << xmin << "\t" << ymin << "\t"
                 << xmax << "\t" << ymax << "\t" << confidence << "\n";
//            rectangle(frame.mat, Point(xmin, ymin), Point(xmax, ymax),
//                      label_color[label], 1.5, cv::LINE_AA, 0);
            int line_length = 0;
            int l0 = xmax - xmin;
            int l1 = ymax - ymin;

            if (l0 > l1) line_length = l1;
            else line_length = l0;

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
//            rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled

            rectangle(frame.mat, Point(xmin, ymin) - Point(0, textSize.height), Point(xmin, ymin) + Point(textSize.width, 0), label_color[label], cv::FILLED);
            putText(frame.mat, show_text, Point(xmin, ymin),   cv::FONT_HERSHEY_DUPLEX, 0.3, Scalar(225, 255, 255), 1, cv::LINE_AA, 0);
//
//            putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
//            int fps = ((float)total_step) / ((float)step);
//            std::string show_fps = "FPS: " + std::to_string(fps);
//            textSize = getTextSize(show_fps, cv::FONT_HERSHEY_DUPLEX, 1, 1, &baseline);
//            putText(frame.mat, show_fps, Point(frame.mat.rows, 0) + Point(-textSize.width, textSize.height),   cv::FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 1, cv::LINE_AA, 0);

        }
//        Mat img_show;
//        hconcat(img, frame.mat, img_show);
//        cv::imshow("result", frame.mat);
//        imwrite(frame.channel_id + "_result.jpg", frame.mat);
//        cv::waitKey(1);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEMO))
        << "dpu queue size " << queue_out_->size();
    while (!queue_out_->push(frame, std::chrono::milliseconds(500))) {
      if (is_stopped()) {
        return -1;
      }
    }
    return 0;
  }

  virtual std::string name() override { return std::string("DPU-") + suffix_; }
  unique_ptr<vitis::ai::DpuTask> model_;
  queue_t* queue_in_;
  queue_t* queue_out_;
  std::string suffix_;
};

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

class Queue_Retuen {
public:
  unique_ptr<queue_t> input_queue;
  unique_ptr<queue_t> output_queue;
  unique_ptr<DpuThread> dpu_thread;
  unique_ptr<SortingThread> sorting_thread;
};

unique_ptr<Queue_Retuen> setup(){

    auto decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
    auto sorting_queue =
        std::unique_ptr<queue_t>(new queue_t(500 * g_num_of_threads));
    auto dpu_thread =std::unique_ptr<DpuThread>(
        new DpuThread(
          decode_queue.get(), sorting_queue.get(), std::to_string(0)));
    auto gui_queue = std::unique_ptr<queue_t>{new queue_t{10}};
    auto sorting_thread = std::unique_ptr<SortingThread>(
        new SortingThread(sorting_queue.get(), gui_queue.get(), std::to_string(0)));

//    auto kernel_name = "yolov5s_191.xmodel";
//    auto task = vitis::ai::DpuTask::create(kernel_name);
//    auto input_tensor = task->getInputTensor(0u);
//    int width = input_tensor[0].width;
//    int height = input_tensor[0].height;
//    auto handle = pp_kernel_init("/media/sd-mmcblk0p1/dpu.xclbin", "pp_pipeline_accel",0);
//    using model_t = typename decltype(task)::element_type;
//
//    auto channel_id = 0;
//    auto decode_queue = std::unique_ptr<queue_t>{new queue_t{5}};
//    auto sorting_queue =
//        std::unique_ptr<queue_t>(new queue_t(500 * g_num_of_threads));
//    auto gui_queue = std::unique_ptr<queue_t>{new queue_t{10}};
//    auto sorting_thread = std::unique_ptr<SortingThread>(
//        new SortingThread(sorting_queue.get(), gui_queue.get(), std::to_string(0)));
    MyThread::start_all();

    auto my_queue = make_unique<Queue_Retuen>();
    my_queue->input_queue = move(decode_queue);
    my_queue->output_queue = move(gui_queue);
    my_queue->dpu_thread = move(dpu_thread);
    my_queue->sorting_thread = move(sorting_thread);

    return move(my_queue);
}