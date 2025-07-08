package com.whispertflite.engine;

import android.content.Context;
import android.util.Log;

// import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
// import com.google.android.gms.tflite.gpu.support.TfLiteGpu;
// import com.google.android.gms.tflite.java.TfLite;
import com.whispertflite.utils.WaveUtil;
import com.whispertflite.utils.WhisperUtil;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class WhisperEngineJava implements WhisperEngine {
    private final String TAG = "WhisperEngineJava";
    private final WhisperUtil mWhisperUtil = new WhisperUtil();

    private final Context mContext;
    private boolean mIsInitialized = false;
    private Interpreter mInterpreter = null;
    private GpuDelegate gpuDelegate;
    private boolean isGpuSupported = false;

    public WhisperEngineJava(Context context) {
        mContext = context;
    }

    @Override
    public boolean isInitialized() {
        return mIsInitialized;
    }

    @Override
    public boolean initialize(String modelPath, String vocabPath, boolean multilingual) throws IOException {
        // Load model
        loadModel(modelPath);
        Log.d(TAG, "Model is loaded..." + modelPath);

        // Load filters and vocab
        boolean ret = mWhisperUtil.loadFiltersAndVocab(multilingual, vocabPath);
        if (ret) {
            mIsInitialized = true;
            Log.d(TAG, "Filters and Vocab are loaded..." + vocabPath);
        } else {
            mIsInitialized = false;
            Log.d(TAG, "Failed to load Filters and Vocab...");
        }

        return mIsInitialized;
    }

    // Unload the model by closing the interpreter
    @Override
    public void deinitialize() {
        if (mInterpreter != null) {
            mInterpreter.close();
            mInterpreter = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
    }

    @Override
    public String transcribeFile(String wavePath) {
        // Calculate Mel spectrogram
        Log.d(TAG, "Calculating Mel spectrogram...");
        float[] melSpectrogram = getMelSpectrogram(wavePath);
        Log.d(TAG, "Mel spectrogram is calculated...!");

        // Perform inference
        String result = runInference(melSpectrogram);
        Log.d(TAG, "Inference is executed...!");

        return result;
    }

    @Override
    public String transcribeBuffer(float[] samples) {
        return null;
    }
    
    // Method to check if GPU acceleration is being used
    public boolean isUsingGpuAcceleration() {
        return isGpuSupported;
    }
    
    // Method to get acceleration type being used
    public String getAccelerationType() {
        if (isGpuSupported) {
            return "GPU";
        } else {
            return "CPU + XNNPACK";
        }
    }

    // Load TFLite model with GPU acceleration
    private void loadModel(String modelPath) throws IOException {
        FileInputStream fileInputStream = new FileInputStream(modelPath);
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = 0;
        long declaredLength = fileChannel.size();
        ByteBuffer tfliteModel = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

        // Set the number of threads for inference
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(Runtime.getRuntime().availableProcessors());
        
        // Try to initialize GPU acceleration
        initializeGpuAcceleration(options, modelPath);
        
        // Enable XNNPACK for CPU optimization as fallback
        options.setUseXNNPACK(true);
        
        // Allow FP16 precision for better performance
        options.setAllowFp16PrecisionForFp32(true);
        
        mInterpreter = new Interpreter(tfliteModel, options);
        
        Log.d(TAG, "Model loaded with " + (isGpuSupported ? "GPU" : "CPU") + " acceleration");
    }
    
    private void initializeGpuAcceleration(Interpreter.Options options, String modelPath) {
        try {
            // Check GPU compatibility using CompatibilityList
            CompatibilityList compatList = new CompatibilityList();
            
            if (compatList.isDelegateSupportedOnThisDevice()) {
                Log.d(TAG, "GPU delegate is supported on this device");
                
                // Create GPU delegate with optimized settings
                GpuDelegate.Options gpuOptions = compatList.getBestOptionsForThisDevice();
                
                // Configure GPU delegate for optimal performance
                gpuOptions.setPrecisionLossAllowed(true); // Allow FP16 for faster inference
                gpuOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
                gpuOptions.setSerializationDir(mContext.getCacheDir()); // Enable serialization for faster startup
                gpuOptions.setModelToken(modelPath.hashCode()); // Use model path hash as token
                
                gpuDelegate = new GpuDelegate(gpuOptions);
                options.addDelegate(gpuDelegate);
                
                isGpuSupported = true;
                Log.d(TAG, "GPU delegate initialized successfully");
                
            } else {
                Log.d(TAG, "GPU delegate not supported on this device, falling back to CPU");
                isGpuSupported = false;
            }
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize GPU delegate: " + e.getMessage());
            isGpuSupported = false;
            
            // Clean up GPU delegate if initialization failed
            if (gpuDelegate != null) {
                gpuDelegate.close();
                gpuDelegate = null;
            }
        }
        
        // Fallback to NNAPI if GPU is not available (for newer Android devices)
        if (!isGpuSupported) {
            try {
                NnApiDelegate nnapiDelegate = new NnApiDelegate();
                options.addDelegate(nnapiDelegate);
                Log.d(TAG, "NNAPI delegate initialized as fallback");
            } catch (Exception e) {
                Log.d(TAG, "NNAPI delegate also not available, using CPU only");
            }
        }
    }

    private float[] getMelSpectrogram(String wavePath) {
        // Get samples in PCM_FLOAT format
        float[] samples = WaveUtil.getSamples(wavePath);

        int fixedInputSize = WhisperUtil.WHISPER_SAMPLE_RATE * WhisperUtil.WHISPER_CHUNK_SIZE;
        float[] inputSamples = new float[fixedInputSize];
        int copyLength = Math.min(samples.length, fixedInputSize);
        System.arraycopy(samples, 0, inputSamples, 0, copyLength);

        int cores = Runtime.getRuntime().availableProcessors();
        return mWhisperUtil.getMelSpectrogram(inputSamples, inputSamples.length, cores);
    }

    private String runInference(float[] inputData) {
        // Create input tensor
        Tensor inputTensor = mInterpreter.getInputTensor(0);
        TensorBuffer inputBuffer = TensorBuffer.createFixedSize(inputTensor.shape(), inputTensor.dataType());
//        printTensorDump("Input Tensor Dump ===>", inputTensor);

        // Create output tensor
        Tensor outputTensor = mInterpreter.getOutputTensor(0);
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), DataType.FLOAT32);
//        printTensorDump("Output Tensor Dump ===>", outputTensor);

        // Load input data
        int inputSize = inputTensor.shape()[0] * inputTensor.shape()[1] * inputTensor.shape()[2] * Float.BYTES;
        ByteBuffer inputBuf = ByteBuffer.allocateDirect(inputSize);
        inputBuf.order(ByteOrder.nativeOrder());
        for (float input : inputData) {
            inputBuf.putFloat(input);
        }

        // To test mel data as a input directly
//        try {
//            byte[] bytes = Files.readAllBytes(Paths.get("/data/user/0/com.example.tfliteaudio/files/mel_spectrogram.bin"));
//            inputBuf = ByteBuffer.wrap(bytes);
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }

        inputBuffer.loadBuffer(inputBuf);

//        Log.d(TAG, "Before inference...");
        // Run inference
        mInterpreter.run(inputBuffer.getBuffer(), outputBuffer.getBuffer());
//        Log.d(TAG, "After inference...");

        // Retrieve the results
        int outputLen = outputBuffer.getIntArray().length;
        Log.d(TAG, "output_len: " + outputLen);
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < outputLen; i++) {
            int token = outputBuffer.getBuffer().getInt();
            if (token == mWhisperUtil.getTokenEOT())
                break;

            // Get word for token and Skip additional token
            if (token < mWhisperUtil.getTokenEOT()) {
                String word = mWhisperUtil.getWordFromToken(token);
                //Log.d(TAG, "Adding token: " + token + ", word: " + word);
                result.append(word);
            } else {
                if (token == mWhisperUtil.getTokenTranscribe())
                    Log.d(TAG, "It is Transcription...");

                if (token == mWhisperUtil.getTokenTranslate())
                    Log.d(TAG, "It is Translation...");

                String word = mWhisperUtil.getWordFromToken(token);
                Log.d(TAG, "Skipping token: " + token + ", word: " + word);
            }
        }

        return result.toString();
    }

    private void printTensorDump(String message, Tensor tensor) {
        Log.d(TAG,"Output Tensor Dump ===>");
        Log.d(TAG, "  shape.length: " + tensor.shape().length);
        for (int i = 0; i < tensor.shape().length; i++)
            Log.d(TAG, "    shape[" + i + "]: " + tensor.shape()[i]);
        Log.d(TAG, "  dataType: " + tensor.dataType());
        Log.d(TAG, "  name: " + tensor.name());
        Log.d(TAG, "  numBytes: " + tensor.numBytes());
        Log.d(TAG, "  index: " + tensor.index());
        Log.d(TAG, "  numDimensions: " + tensor.numDimensions());
        Log.d(TAG, "  numElements: " + tensor.numElements());
        Log.d(TAG, "  shapeSignature.length: " + tensor.shapeSignature().length);
        Log.d(TAG, "  quantizationParams.getScale: " + tensor.quantizationParams().getScale());
        Log.d(TAG, "  quantizationParams.getZeroPoint: " + tensor.quantizationParams().getZeroPoint());
        Log.d(TAG, "==================================================================");
    }
}
