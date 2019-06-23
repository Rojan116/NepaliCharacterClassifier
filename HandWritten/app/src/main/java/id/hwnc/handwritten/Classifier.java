package id.hwnc.handwritten;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class Classifier  extends AppCompatActivity {

    private static final String TAG ="" ;
    // Output array [batch_size, 36]
    private float[][] nepaliOutput = null;

    // options for model interpreter
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    // tflite graph
    private Interpreter tflite;
    // holds all the possible labels for model
    private List<String> labelList;
    // holds the selected image data as bytes
    private ByteBuffer imgData = null;
    // holds the probabilities of each label for non-quantized graphs
    private float[][] labelProbArray = null;
    // holds the probabilities of each label for quantized graphs
    private byte[][] labelProbArrayB = null;
    // array that holds the labels with the highest probabilities
    private String[] topLables = null;
    // array that holds the highest probabilities
    private String[] topConfidence = null;


    // selected classifier information received from extras
    private String chosen ="model.tflite";
    private boolean quant;

    // input image dimensions for the Inception Model
    private int DIM_IMG_SIZE_X = 32;
    private int DIM_IMG_SIZE_Y = 32;
    private int DIM_PIXEL_SIZE = 1;

    // int array to hold image data
    //private int[] intValues;

    // activity elements

    private Button btn_class;
    private Button btn_clear;
    private TextView tfRes;
    private TextView tfConf;



    
    
    
    // initialize array that holds image data
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];




    public Classifier(Activity activity) {
        try {
            tflite = new Interpreter(loadModelFile(activity));



            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
                    4 * 1 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
            inputBuffer.order(ByteOrder.nativeOrder());
            nepaliOutput = new float[1][36];
           Log.d(TAG, "Created a Tensorflow Lite Nepali Classifier.");
        } catch (IOException e) {
            Log.e(TAG, "IOException loading the tflite file");
        }
    }


    
    
    // loads the labels from the label txt file in assets into a string array
    public List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();

        return labelList;
    }

    // loads tflite grapg from file
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(chosen);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }




    // converts bitmap to byte array which is passed in the tflite graph
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                // if quantized, convert each rgb value to a byte, otherwise to a float
                if (quant) {
                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) (val & 0xFF));
                } else {

                    imgData.putFloat((((val) & 0xFF)));
                }

            }
        }
    }



    /**
     * Run the TFLite model
     */
    protected void runInference() {
        tflite.run(imgData, nepaliOutput);
    }



    /**
     * Classifies the number with the mnist model.
     *
     * @param bitmap
     * @return the identified number
     */
    public int classify(Bitmap bitmap) {
        if (tflite == null) {

        }
        convertBitmapToByteBuffer(bitmap);
        runInference();
        return postprocess();
    }

    /**
     * Go through the output and find the number that was identified.
     *
     * @return the number that was identified (returns -1 if one wasn't found)
     */
    private int postprocess() {
        for (int i = 0; i < nepaliOutput[0].length; i++) {
            float value = nepaliOutput[0][i];
            if (value == 1f) {
                return i;
            }
        }
        return -1;
    }




}
