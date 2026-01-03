#include <cmath> 
#include<iostream>
#include <cstdlib> 
#include <iostream>

#include "pico/stdlib.h"

#include "model.h"
#include "inference.h"
#include "model_settings.h"
#include "model_data.h"

using namespace std; 

#define HALT_CORE_1() while (1) { tight_loop_contents(); }




int count_digits(int number) {
    if (number == 0) {
        return 1; // Special case for 0, which has 1 digit
    }

    number = abs(number); // Handle negative numbers
    return std::floor(std::log10(number)) + 1;
}

void inference_test(void) 
{
    Model ml_model;

    if (!ml_model.setup()) {
        printf("Failed to initialize ML model!\n");
        HALT_CORE_1();
    }
    printf("Model initialized\n");
    
    uint8_t* test_image_input = ml_model.input_data();
    if (test_image_input == nullptr) {
        printf("Cannot set input\n");
        HALT_CORE_1();
    }
    
    int byte_size = ml_model.byte_size();
    if (!byte_size) {
        printf("Byte size not found\n");
        HALT_CORE_1();
    }

    while (true) {
        sleep_ms(10000);
    }
    
}