package com.example.javacv_opencv_tflite;

import org.opencv.core.Mat;

public class MatNumberUtils {
    private Integer number;
    private Mat iamge;

    public void setNumber(Integer number) {
        this.number = number;
    }

    public void setIamge(Mat iamge) {
        this.iamge = iamge;
    }

    public Integer getNumber() {
        return number;
    }

    public Mat getIamge() {
        return iamge;
    }
}
