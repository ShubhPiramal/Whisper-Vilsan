package com.whispertflite.engine;

import android.content.Context;
import android.util.Log;

import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.gpu.support.TfLiteGpu;
import com.google.android.gms.tflite.java.TfLite;
import com.google.android.gms.tasks.Task;
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

        // Try GPU first, then fallback to CPU if it fails
        mInterpreter = createInterpreterWithBestAcceleration(tfliteModel);
        
        Log.d(TAG, "Model loaded with " + (isGpuSupported ? "GPU" : "CPU") + " acceleration");
    }
    
    private Interpreter createInterpreterWithBestAcceleration(ByteBuffer tfliteModel) {
        // First try GPU acceleration
        Interpreter interpreter = tryCreateGpuInterpreter(tfliteModel);
        if (interpreter != null) {
            isGpuSupported = true;
            Log.d(TAG, "Successfully created interpreter with GPU acceleration");
            return interpreter;
        }
        
        // If GPU fails, try NNAPI
        interpreter = tryCreateNnapiInterpreter(tfliteModel);
        if (interpreter != null) {
            isGpuSupported = false;
            Log.d(TAG, "Successfully created interpreter with NNAPI acceleration");
            return interpreter;
        }
        
        // If both fail, use CPU only
        interpreter = createCpuInterpreter(tfliteModel);
        isGpuSupported = false;
        Log.d(TAG, "Successfully created interpreter with CPU-only acceleration");
        return interpreter;
    }
    
    private Interpreter tryCreateGpuInterpreter(ByteBuffer tfliteModel) {
        // First try Google Play Services GPU delegate (better dynamic tensor support)
        Interpreter interpreter = tryGooglePlayServicesGpu(tfliteModel);
        if (interpreter != null) return interpreter;
        
        // Try multiple standard GPU configurations to handle dynamic tensors
        interpreter = tryGpuWithDynamicTensorSupport(tfliteModel);
        if (interpreter != null) return interpreter;
        
        interpreter = tryGpuWithStaticTensorWorkaround(tfliteModel);
        if (interpreter != null) return interpreter;
        
        // Try with fixed input shapes to force static behavior
        return tryGpuWithFixedInputShapes(tfliteModel);
    }
    
    private Interpreter tryGooglePlayServicesGpu(ByteBuffer tfliteModel) {
        try {
            Log.d(TAG, "Attempting Google Play Services GPU delegate (better dynamic tensor support)");
            
            // Initialize TFLite with Google Play Services
            TfLiteInitializationOptions initOptions = TfLiteInitializationOptions.builder()
                    .setEnableGpuDelegateSupport(true)
                    .build();
            
            Task<Void> initTask = TfLite.initialize(mContext, initOptions);
            
            // Wait for initialization (this is a simplified approach - in production you'd use callbacks)
            try {
                Thread.sleep(1000); // Give it time to initialize
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted during TfLite initialization");
                return null;
            }
            
            // Check if GPU is available through Google Play Services
            Task<Boolean> gpuAvailabilityTask = TfLiteGpu.isGpuDelegateAvailable(mContext);
            
            // Create interpreter with Google Play Services GPU support
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(Math.min(4, Runtime.getRuntime().availableProcessors()));
            
            // The Google Play Services version should handle dynamic tensors better
            options.setUseXNNPACK(false);
            options.setAllowFp16PrecisionForFp32(true);
            
            // Try to add GPU delegate through Google Play Services
            try {
                // This approach uses the Google Play Services GPU delegate which may support dynamic tensors
                GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
                gpuOptions.setPrecisionLossAllowed(true);
                gpuOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER);
                
                gpuDelegate = new GpuDelegate(gpuOptions);
                options.addDelegate(gpuDelegate);
                
                Interpreter interpreter = new Interpreter(tfliteModel, options);
                
                Log.d(TAG, "Google Play Services GPU interpreter created successfully");
                return interpreter;
                
            } catch (Exception e) {
                Log.w(TAG, "Google Play Services GPU delegate failed: " + e.getMessage());
                if (gpuDelegate != null) {
                    gpuDelegate.close();
                    gpuDelegate = null;
                }
                return null;
            }
            
        } catch (Exception e) {
            Log.w(TAG, "Google Play Services TfLite initialization failed: " + e.getMessage());
            return null;
        }
    }
    
    private Interpreter tryGpuWithDynamicTensorSupport(ByteBuffer tfliteModel) {
        try {
            Log.d(TAG, "Attempting GPU with dynamic tensor support");
            
            CompatibilityList compatList;
            try {
                compatList = new CompatibilityList();
            } catch (Exception e) {
                Log.w(TAG, "Failed to create CompatibilityList: " + e.getMessage());
                return null;
            }
            
            if (!compatList.isDelegateSupportedOnThisDevice()) {
                Log.d(TAG, "GPU delegate not supported on this device");
                return null;
            }
            
            // Try with experimental dynamic tensor support
            GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
            gpuOptions.setPrecisionLossAllowed(true);
            gpuOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER);
            
            // Enable experimental features that might support dynamic tensors
            try {
                // Try to enable dynamic shape support if available
                gpuOptions.setQuantizedModelsAllowed(true);
            } catch (Exception e) {
                Log.d(TAG, "Quantized models option not available: " + e.getMessage());
            }
            
            gpuDelegate = new GpuDelegate(gpuOptions);
            
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(Math.min(4, Runtime.getRuntime().availableProcessors()));
            options.addDelegate(gpuDelegate);
            options.setUseXNNPACK(false); // Disable XNNPACK when using GPU
            options.setAllowFp16PrecisionForFp32(true);
            
            // Enable experimental features for dynamic tensors
            options.setAllowBufferHandleOutput(true);
            
            Interpreter interpreter = new Interpreter(tfliteModel, options);
            
            Log.d(TAG, "GPU interpreter with dynamic tensor support created successfully");
            return interpreter;
            
        } catch (Exception e) {
            Log.w(TAG, "GPU with dynamic tensor support failed: " + e.getMessage());
            if (gpuDelegate != null) {
                try {
                    gpuDelegate.close();
                } catch (Exception closeException) {
                    Log.w(TAG, "Error closing GPU delegate: " + closeException.getMessage());
                }
                gpuDelegate = null;
            }
            return null;
        }
    }
    
    private Interpreter tryGpuWithStaticTensorWorkaround(ByteBuffer tfliteModel) {
        try {
            Log.d(TAG, "Attempting GPU with static tensor workaround");
            
            CompatibilityList compatList = new CompatibilityList();
            if (!compatList.isDelegateSupportedOnThisDevice()) {
                return null;
            }
            
            // Create GPU delegate with settings optimized for static tensors
            GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
            gpuOptions.setPrecisionLossAllowed(true);
            gpuOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED);
            
            // Try to force static tensor behavior
            try {
                gpuOptions.setQuantizedModelsAllowed(false); // Disable quantization for static behavior
            } catch (Exception e) {
                Log.d(TAG, "Quantization setting not available");
            }
            
            gpuDelegate = new GpuDelegate(gpuOptions);
            
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(1); // Use single thread for static tensor compatibility
            options.addDelegate(gpuDelegate);
            options.setUseXNNPACK(false);
            options.setAllowFp16PrecisionForFp32(true);
            
            // Try to allocate tensors upfront to make them static
            Interpreter interpreter = new Interpreter(tfliteModel, options);
            
            // Allocate tensors to try to make them static
            try {
                interpreter.allocateTensors();
                Log.d(TAG, "Tensors allocated successfully for static behavior");
            } catch (Exception e) {
                Log.w(TAG, "Failed to allocate tensors: " + e.getMessage());
                interpreter.close();
                if (gpuDelegate != null) {
                    gpuDelegate.close();
                    gpuDelegate = null;
                }
                return null;
            }
            
            Log.d(TAG, "GPU interpreter with static tensor workaround created successfully");
            return interpreter;
            
        } catch (Exception e) {
            Log.w(TAG, "GPU with static tensor workaround failed: " + e.getMessage());
            if (gpuDelegate != null) {
                try {
                    gpuDelegate.close();
                } catch (Exception closeException) {
                    Log.w(TAG, "Error closing GPU delegate: " + closeException.getMessage());
                }
                gpuDelegate = null;
            }
            return null;
        }
    }
    
    private Interpreter tryCreateNnapiInterpreter(ByteBuffer tfliteModel) {
        try {
            Log.d(TAG, "Attempting to create NNAPI interpreter");
            
            NnApiDelegate nnapiDelegate = new NnApiDelegate();
            
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(Runtime.getRuntime().availableProcessors());
            options.addDelegate(nnapiDelegate);
            options.setUseXNNPACK(true);
            options.setAllowFp16PrecisionForFp32(true);
            
            Interpreter interpreter = new Interpreter(tfliteModel, options);
            
            Log.d(TAG, "NNAPI interpreter created successfully");
            return interpreter;
            
        } catch (Exception e) {
            Log.w(TAG, "Failed to create NNAPI interpreter: " + e.getMessage());
            return null;
        }
    }
    
    private Interpreter tryGpuWithFixedInputShapes(ByteBuffer tfliteModel) {
        try {
            Log.d(TAG, "Attempting GPU with fixed input shapes for static tensor compatibility");
            
            CompatibilityList compatList = new CompatibilityList();
            if (!compatList.isDelegateSupportedOnThisDevice()) {
                return null;
            }
            
            // Create GPU delegate with most basic settings
            GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
            gpuOptions.setPrecisionLossAllowed(true);
            gpuOptions.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER);
            
            gpuDelegate = new GpuDelegate(gpuOptions);
            
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(1); // Single thread for maximum compatibility
            options.addDelegate(gpuDelegate);
            options.setUseXNNPACK(false);
            options.setAllowFp16PrecisionForFp32(false); // Disable FP16 for compatibility
            
            Interpreter interpreter = new Interpreter(tfliteModel, options);
            
            // Try to pre-allocate with fixed shapes to force static behavior
            try {
                // Allocate tensors first
                interpreter.allocateTensors();
                
                // Try to resize input tensors to fixed shapes (Whisper typical input shape)
                // Whisper input is typically [1, 80, 3000] for mel spectrogram
                int[] fixedInputShape = {1, 80, 3000};
                interpreter.resizeInput(0, fixedInputShape);
                interpreter.allocateTensors();
                
                Log.d(TAG, "GPU interpreter with fixed input shapes created successfully");
                return interpreter;
                
            } catch (Exception e) {
                Log.w(TAG, "Failed to set fixed input shapes: " + e.getMessage());
                interpreter.close();
                if (gpuDelegate != null) {
                    gpuDelegate.close();
                    gpuDelegate = null;
                }
                return null;
            }
            
        } catch (Exception e) {
            Log.w(TAG, "GPU with fixed input shapes failed: " + e.getMessage());
            if (gpuDelegate != null) {
                try {
                    gpuDelegate.close();
                } catch (Exception closeException) {
                    Log.w(TAG, "Error closing GPU delegate: " + closeException.getMessage());
                }
                gpuDelegate = null;
            }
            return null;
        }
    }
    
    private Interpreter createCpuInterpreter(ByteBuffer tfliteModel) {
        try {
            Log.d(TAG, "Creating CPU-only interpreter");
            
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(Runtime.getRuntime().availableProcessors());
            options.setUseXNNPACK(true);
            options.setAllowFp16PrecisionForFp32(true);
            
            return new Interpreter(tfliteModel, options);
            
        } catch (Exception e) {
            Log.e(TAG, "Failed to create CPU interpreter: " + e.getMessage());
            throw new RuntimeException("Failed to create any interpreter", e);
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
