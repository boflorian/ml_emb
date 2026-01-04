#include <cstdarg>
#include <cstdio>

#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "model.h"
#include "model_settings.h"
#include "model_data.h" // Ensure this header defines `model_data` and `model_data_len`

namespace {
// Simple reporter that forces messages to printf so we see why setup fails.
class PicoErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    vprintf(format, args);
    printf("\n");
    return 0;
  }
};
}  // namespace

Model::Model() :
    model(nullptr),
    interpreter(nullptr),
    input(nullptr),
    error_reporter(nullptr)
{
}

Model::~Model()
{
    if (interpreter != NULL) {
        delete interpreter;
        interpreter = NULL;
    }
    if (input != NULL) {
        delete input;
        input = NULL;
    }
}

int Model::setup() 
{
    static PicoErrorReporter pico_error_reporter;
    error_reporter = &pico_error_reporter;

    // Use model_data_len directly instead of sizeof
    extern const unsigned int model_data_len;

    //printf("Model::setup start\n");
    //printf("Model blob length: %u\n", model_data_len);
    //printf("Model blob first 16 bytes: ");
    //for (int i = 0; i < 16; ++i) printf("%02X ", model_data[i]);
    //printf("\n");
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return 0;
    }

    static tflite::MicroMutableOpResolver<50> micro_op_resolver; // Allow more ops
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddLeakyRelu();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddExpandDims();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddBatchMatMul();
    micro_op_resolver.AddPack();        // Used by GlobalAveragePooling / axis ops
    micro_op_resolver.AddLogistic();    // For LayerNormalization epsilon path
    micro_op_resolver.AddLogSoftmax();  // For safety if present in converted head
    micro_op_resolver.AddSub();
    micro_op_resolver.AddDiv();
    micro_op_resolver.AddRsqrt();
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPad();
    micro_op_resolver.AddSpaceToBatchNd();
    micro_op_resolver.AddBatchToSpaceNd();
    micro_op_resolver.AddSquare();
    micro_op_resolver.AddSquaredDifference();
    micro_op_resolver.AddMaximum();
    micro_op_resolver.AddMinimum();
    micro_op_resolver.AddCast();
    micro_op_resolver.AddNeg();

    static uint8_t tensor_arena[arena_size];
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, arena_size);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return 0;
    }

    input = interpreter->input(0);
    if (input) {
        printf("Input type=%d dims:", input->type);
        for (int i = 0; i < input->dims->size; ++i) {
            printf(" %d", input->dims->data[i]);
        }
        printf(" bytes=%d scale=%f zero_point=%d\n", input->bytes, input->params.scale, input->params.zero_point);
    }

    printf("Model::setup success\n");
    return 1;
}

uint8_t* Model::input_data() {
  if (input == nullptr) {
    return nullptr;
  }
  return input->data.uint8;
}

int Model::byte_size() {
  if (input == nullptr) {
    return 0;
  }
  return input->bytes;
}

float Model::input_scale() {
  if (input == nullptr) {
    return 1.0f;
  }
  return input->params.scale;
}

int Model::input_zero_point() {
  if (input == nullptr) {
    return 0;
  }
  return input->params.zero_point;
}

float* Model::output_data() {
    if (interpreter == nullptr) return nullptr;
    TfLiteTensor* output = interpreter->output(0);
    return output->data.f;
}

int Model::predict()
{
  printf("Invocation started\n");

  if (interpreter->Invoke() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return -1;
  }

  printf("Invocation finished\n");

  TfLiteTensor* output = interpreter->output(0);

  // Debug: print output tensor info
  printf("Output type=%d dims:", output->type);
  for (int i = 0; i < output->dims->size; ++i) {
      printf(" %d", output->dims->data[i]);
  }
  printf("\n");

  // Get number of classes (last dimension)
  int num_classes = output->dims->data[output->dims->size - 1];
  
  int result = 0;
  
  // Bias correction to counteract model bias towards certain classes
  // Positive values boost a class, negative values penalize
  // Classes: 0=negative, 1=ring, 2=slope, 3=wave
  // Tune these values based on observed bias
  static const int output_bias[4] = {
      50,    // boost negative
      100,    // boost ring slightly
      80,   // LARGE boost for slope (dead neuron workaround)
      -150    // penalize wave (model is biased towards it)
  };
  
  // Check if output is quantized (int8) or float
  if (output->type == kTfLiteInt8) {
      // INT8 quantized output
      int8_t* output_int8 = output->data.int8;
      
      // Apply bias correction and find max
      int corrected_scores[4];
      printf("Output scores (int8): ");
      for (int i = 0; i < num_classes; ++i) {
          printf("%d ", output_int8[i]);
          corrected_scores[i] = (int)output_int8[i] + output_bias[i];
      }
      printf("\n");
      
      // Check if slope neuron appears dead (always near -128)
      // If so, use input variance as a heuristic for slope detection
      bool slope_neuron_dead = (output_int8[2] <= -125);
      
      printf("Bias-corrected scores: ");
      int max_corrected = corrected_scores[0];
      int second_max = -999;
      int second_idx = -1;
      for (int i = 0; i < num_classes; ++i) {
          printf("%d ", corrected_scores[i]);
          if (corrected_scores[i] > max_corrected) {
              second_max = max_corrected;
              second_idx = result;
              max_corrected = corrected_scores[i];
              result = i;
          } else if (corrected_scores[i] > second_max) {
              second_max = corrected_scores[i];
              second_idx = i;
          }
      }
      printf("\n");
      
      // Print confidence info
      int margin = max_corrected - second_max;
      printf("Top: class %d (%d), Second: class %d (%d), Margin: %d\n", 
             result, max_corrected, second_idx, second_max, margin);
      
      // If margin is very small, flag as uncertain
      if (margin < 20) {
          printf("WARNING: Low confidence (margin=%d < 20)\n", margin);
      }
  } else {
      // Float output
      float max_value = output->data.f[0];
      printf("Output scores (float): ");
      for (int i = 0; i < num_classes; ++i) {
          printf("%.4f ", output->data.f[i]);
          if (output->data.f[i] > max_value) {
              max_value = output->data.f[i];
              result = i;
          }
      }
      printf("\n");
  }

  return result;
}
