#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "model.h"
#include "model_settings.h"
#include "model_data.h" // Ensure this header defines `model_data` and `model_data_len`

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
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Use model_data_len directly instead of sizeof
    extern const unsigned int model_data_len;

    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return 0;
    }

    static tflite::MicroMutableOpResolver<5> micro_op_resolver; // Adjust size as needed
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddSoftmax();

    static uint8_t tensor_arena[arena_size];
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, arena_size);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return 0;
    }

    input = interpreter->input(0);

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