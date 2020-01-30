package com.developndesign.mlkit4facedetectiion;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionPoint;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceContour;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceLandmark;
import com.theartofdev.edmodo.cropper.CropImage;
import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    FirebaseVisionImage image; // preparing the input image
    TextView textView; // Displaying the face detection data for the input image
    Button button; // To select the image from device
    ImageView imageView; //To display the selected image

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView = findViewById(R.id.text);
        button = findViewById(R.id.selectImage);
        imageView = findViewById(R.id.image);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                CropImage.activity().start(MainActivity.this);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);
            if (resultCode == RESULT_OK) {
                if (result != null) {
                    Uri uri = result.getUri(); //path of image in phone
                    imageView.setImageURI(uri); //set image in imageview
                    textView.setText(""); //so that previous text don't get append with new one
                    detectFaceFromImage(uri);
                }
            }
        }
    }

    private void detectFaceFromImage(Uri uri) {
        try {
            image = FirebaseVisionImage.fromFilePath(MainActivity.this, uri);
            FirebaseVisionFaceDetectorOptions highAccuracyOpts =
                    new FirebaseVisionFaceDetectorOptions.Builder()
                            .setPerformanceMode(FirebaseVisionFaceDetectorOptions.ACCURATE)
                            .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
                            .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
                            .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                            .build();
            FirebaseVisionFaceDetector detector = FirebaseVision.getInstance()
                    .getVisionFaceDetector(highAccuracyOpts);

            detector.detectInImage(image)
                    .addOnSuccessListener(new OnSuccessListener<List<FirebaseVisionFace>>() {
                        @Override
                        public void onSuccess(List<FirebaseVisionFace> faces) {
                            for (FirebaseVisionFace face : faces) {
                                Rect bounds = face.getBoundingBox();
                                textView.append("Bounding Polygon "+ "("+bounds.centerX()+","+bounds.centerY()+")"+"\n\n");
                                float rotY = face.getHeadEulerAngleY();  // Head is rotated to the right rotY degrees
                                float rotZ = face.getHeadEulerAngleZ();  // Head is tilted sideways rotZ degrees
                                textView.append("Angles of rotation " + "Y:"+rotY+","+ "Z: "+rotZ+ "\n\n");
                                // If landmark detection was enabled (mouth, ears, eyes, cheeks, and
                                // nose available):
                                // If face tracking was enabled:
                                if (face.getTrackingId() != FirebaseVisionFace.INVALID_ID) {
                                    int id = face.getTrackingId();
                                    textView.append("id: " + id + "\n\n");
                                }
                                FirebaseVisionFaceLandmark leftEar = face.getLandmark(FirebaseVisionFaceLandmark.LEFT_EAR);
                                if (leftEar != null) {
                                    FirebaseVisionPoint leftEarPos = leftEar.getPosition();
                                    textView.append("LeftEarPos: " + "("+leftEarPos.getX()+"," + leftEarPos.getY()+")"+"\n\n");
                                }
                                FirebaseVisionFaceLandmark rightEar = face.getLandmark(FirebaseVisionFaceLandmark.RIGHT_EAR);
                                if (rightEar != null) {
                                    FirebaseVisionPoint rightEarPos = rightEar.getPosition();
                                    textView.append("RightEarPos: " + "("+rightEarPos.getX()+","+rightEarPos.getY() +")"+ "\n\n");
                                }

                                // If contour detection was enabled:
                                List<FirebaseVisionPoint> leftEyeContour =
                                        face.getContour(FirebaseVisionFaceContour.LEFT_EYE).getPoints();
                                List<FirebaseVisionPoint> upperLipBottomContour =
                                        face.getContour(FirebaseVisionFaceContour.UPPER_LIP_BOTTOM).getPoints();

                                // If classification was enabled:
                                if (face.getSmilingProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
                                    float smileProb = face.getSmilingProbability();
                                    textView.append("SmileProbability: " + ("" + smileProb * 100).subSequence(0, 4) + "%" + "\n\n");
                                }
                                if (face.getRightEyeOpenProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
                                    float rightEyeOpenProb = face.getRightEyeOpenProbability();
                                    textView.append("RightEyeOpenProbability: " + ("" + rightEyeOpenProb * 100).subSequence(0, 4) + "%" + "\n\n");
                                }
                                if (face.getLeftEyeOpenProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
                                    float leftEyeOpenProbability = face.getLeftEyeOpenProbability();
                                    textView.append("LeftEyeOpenProbability: " + ("" + leftEyeOpenProbability * 100).subSequence(0, 4) + "%" + "\n\n");
                                }
                            }
                        }
                    })
                    .addOnFailureListener(
                            new OnFailureListener() {
                                @Override
                                public void onFailure(@NonNull Exception e) {
                                    // Task failed with an exception
                                    // ...
                                }
                            });

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}