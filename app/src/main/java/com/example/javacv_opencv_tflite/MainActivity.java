package com.example.javacv_opencv_tflite;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;


import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import static org.bytedeco.javacpp.avutil.AV_PIX_FMT_RGBA;


public class MainActivity extends AppCompatActivity implements View.OnClickListener{
    // 为当前Activity取一个名字 方便调试
    private String TAG = "MainActivity";
    private Activity activity;
    Properties props = new Properties();


    /*
     *  本实例创建时，自动执行的函数
     * */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        boolean status = OpenCVLoader.initDebug();
        activity = this;
        try {
            props.load(getApplicationContext().getAssets().open("config.properties"));
            initializationArg(props);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (status) {
            Log.e(TAG, "onCreate: Succese");
        } else {
            Log.e(TAG, "onCreate: Failed");
        }

        if (Build.VERSION.SDK_INT >= 23){
            CommonUtils.requestPermissions(this);
        }

        // 创建一个Button对象 并指向前面UI界面中的button组件
        Button btn = (Button)findViewById(R.id.test_button);
        // 为id等于test_button的按钮对象添加一个点击事件
        // 继承View.OnClickListener接口
        btn.setOnClickListener(this);

    }
    // 需要根据配置文件重新初始化算法参数
    public void initializationArg(Properties props){
        timeF_switch_bg = Integer.parseInt(props.getProperty("timeF_switch_bg"));
        tmp_car_category = Integer.parseInt(props.getProperty("tmp_car_category"));
        last_car_category = Integer.parseInt(props.getProperty("last_car_category"));
        car_state_number = Integer.parseInt(props.getProperty("car_state_number"));
        image_sim_number = Integer.parseInt(props.getProperty("image_sim_number"));
        // 必须在OpenCV初始化完毕后才不会报错
        last_image = new Mat();

    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    /*
     * 继承View.OnClickListener接口后 出现的需要重写的方法
     * */
    @Override
    public void onClick(View v) {
        // 根据用户点击的不同按钮 执行不同的逻辑方法
        switch (v.getId()) {
            case R.id.test_button:
                try {
                    readLocalVedio();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                break;
            case R.id.imageView:
                // todo...
        }
    }
    private FFmpegFrameGrabber grabber;
    private AndroidFrameConverter converter;
    private ImageView imageView;
    private Frame frame;
    private Bitmap bmp;
    public void readLocalVedio() throws IOException {
        String file = Environment.getExternalStorageDirectory().toString() + "/pro_dump_data.mp4";
        grabber = FFmpegFrameGrabber.createDefault(file);
        grabber.setImageWidth(1280);
        grabber.setImageHeight(720);
        // 根据id 获取UI界面中的ImageView对象 并把操作结果展示到该对象中
        imageView = (ImageView)this.findViewById(R.id.imageView);
        //为了加快转bitmap这句一定要写
        grabber.setPixelFormat(AV_PIX_FMT_RGBA);
        grabber.start();
        converter = new AndroidFrameConverter();
        frame = grabber.grabImage();
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (frame!= null) {
                    try {
                        frame = grabber.grabImage();
                        bmp = converter.convert(frame);
                        Mat tmp_now_image = new Mat();
                        Utils.bitmapToMat(bmp, tmp_now_image);
                        tmp_now_image = openCVTools.deal_flag(tmp_now_image);
                        matNumberUtils = mainCarBehaviorAnalysis(tmp_now_image);

                        final Bitmap b1 = Bitmap.createBitmap(tmp_now_image.cols(), tmp_now_image.rows(),
                                Bitmap.Config.ARGB_8888);

                        Utils.matToBitmap(matNumberUtils.getIamge(),b1);
                        activity.runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                imageView.setImageBitmap(b1);
                            }
                        });
                    } catch (FrameGrabber.Exception e) {
                        e.printStackTrace();
                    }

                }
            }
        }
        ).start();
    }

    /**
     * openCv 车斗行为识别算法
     *
     */
    private CarBehaviorAnalysisByOpenCv carBehaviorAnalysisByOpenCv = new CarBehaviorAnalysisByOpenCv();
    private OpenCVTools openCVTools = new OpenCVTools();
    private MatNumberUtils matNumberUtils = new MatNumberUtils();
    Mat last_image;  // 上一时刻的图片
    Mat tmp_last_image; // 记录当前时刻图片的中间状态

    // start ======================  OpenCV为主的算法需要的参数 =====================
    int timeF = 3; // 每隔timeF取一张图片
    int timeF_switch_bg = 10;  // 当提取三张图片后 再替换对比底图

    List<Mat> much_catch_image = new ArrayList<>(); // 缓存多张图片 用于服务器上传

    // 数据转换的临时变量
    Integer tmp_car_state = 0; // 以霍夫直线为主 所得到的车辆行为识别结果

    Integer image_sim_number = 10;  // 前后两帧相似度持续大于image_sim_through的上限
    Integer last_car_state = 0;  // 记录车辆上一时刻状态

    Integer car_state_number = 5;  // 记录运输状态的持续次数
    // end ======================  以霍夫直线为主的算法需要的参数 =====================

    // start ======================  以凸包检测为主的算法需要的参数 =====================
    Integer now_image_hull = 0;  // 当前检测区域的凸包数量
    // end ======================  以凸包检测为主的算法需要的参数 =====================


    // start ======================  模型为主的算法需要的参数 =====================
    String model_result = ""; // 记录模型货物分类结果
    int last_car_category = 6; // 默认上一时刻车载类别为篷布
    int tmp_car_category = 6; // 记录车载类别的中间状态
    String textToShow = "";

    public MatNumberUtils mainCarBehaviorAnalysis(Mat tmp_cut_image){
        // ========================= 这里接入车载设备速度
        int tmp_speed = 0;
        int tmp_angle = 0;

        if (timeF <= 0) {
            // 初始化基础参数
            if (last_image.cols() == 0) {
                // 默认为当前货物类别
                last_image = tmp_cut_image;
                tmp_last_image = tmp_cut_image;
            }
            // 计算前后两帧的相似度
            Integer image_sim = openCVTools.split_blok_box_sim(last_image, tmp_cut_image);
            Integer image_sim_through = Integer.parseInt(props.getProperty("image_sim_through"));
            if (image_sim > image_sim_through){
                image_sim_number = Math.min(image_sim_number + 1, Integer.parseInt(props.getProperty("image_sim_number")));
            } else {
                image_sim_number = Math.max(image_sim_number - 1, 0);
            }
            // 凸包检测
            matNumberUtils = openCVTools.other_contours_Hull(tmp_cut_image);
            // 获取凸包数
            now_image_hull = matNumberUtils.getNumber();

            // 结合 陀螺仪 凸包检测 进行车辆行为分析
            tmp_car_state = carBehaviorAnalysisByOpenCv.carBehaviorAnalysisByHull(image_sim_number,now_image_hull,tmp_speed, tmp_angle, props);

            // 防止结果跳变
            if (last_car_state != tmp_car_state){
                car_state_number = Math.max(0,car_state_number -1 );
                if (car_state_number > 0){
                    tmp_car_state = last_car_state;
                } else {
                    last_car_state = tmp_car_state;  // 记录当前时刻车辆状态
                }
            } else {
                car_state_number = Integer.parseInt(props.getProperty("car_state_number"));
            }

            // 相识度对比底片替换 使得last_image与flag相差一定帧数
            if (timeF_switch_bg <= 0) {
                timeF_switch_bg = Integer.parseInt(props.getProperty("timeF_switch_bg"));
                last_image = tmp_last_image;
                tmp_last_image = tmp_cut_image;
            } else {
                timeF_switch_bg = timeF_switch_bg - 1;
            }
        } else{
            timeF = timeF - 1;
            matNumberUtils.setIamge(tmp_cut_image);
        }

        textToShow = "检测结果: " + openCVTools.result_text.get(tmp_car_state) + " \n " +
                "凸包数量: " + now_image_hull + " \n " +
                "相似度 : " + image_sim_number + " \n " +
                "当前角度：" + tmp_angle + " \n " +
                "当前速度：" + tmp_speed + " \n " + model_result;
        Log.i("数据", textToShow);
        return matNumberUtils;
    }
    public void procSrc2Gray(){

        // 按照ARGB_8888的方式读取drawable下的people.jpg图片
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap1 = BitmapFactory.decodeResource(this.getResources(),R.drawable.people);
        // 设置两个Mat对象 存储灰度转换过程中的临时结果
        Mat src = new Mat();
        Mat des = new Mat();

        // 调用Utils工具类，把前面读入的图片转成Mat格式，方便Java识别和操作
        Utils.bitmapToMat(bitmap1,src);
        // 颜色转换
        Imgproc.cvtColor(src, des, Imgproc.COLOR_BGR2GRAY);



        // 把处理的结果转成安卓应用可以识别的bitmap格式
        Utils.matToBitmap(des,bitmap1);
        // 根据id 获取UI界面中的ImageView对象 并把操作结果展示到该对象中
        ImageView imageView = (ImageView)this.findViewById(R.id.imageView);
        imageView.setImageBitmap(bitmap1);
    }

}