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

    printf("Model::setup start\n");
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
    micro_op_resolver.AddMul();         // already added; harmless
    micro_op_resolver.AddAdd();         // already added; harmless
    micro_op_resolver.AddSquare();
    micro_op_resolver.AddSquaredDifference();
    micro_op_resolver.AddMaximum();
    micro_op_resolver.AddMinimum();
    micro_op_resolver.AddCast();

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

  int result = 0;
  float max_value = output->data.f[0];
  for (int i = 1; i < output->dims->data[0]; ++i) {
    if (output->data.f[i] > max_value) {
      max_value = output->data.f[i];
      result = i;
    }
  }

  return result;
}
