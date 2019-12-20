
/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#define NUM_COLOURS 9

__FLAME_GPU_FUNC__ int output_example(xmachine_memory_example_agent* agent, xmachine_message_example_message_list* example_message_messages){

    // 0 output a message.
    add_example_message_message(example_message_messages, agent->id, agent->key, agent->value);

    return 0;
}

/**
 * input_example FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_example_agent. This represents a single agent instance and can be modified directly.
 * @param example_message_messages  example_message_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_example_message_message and get_next_example_message_message functions.
 */
__FLAME_GPU_FUNC__ int input_example(xmachine_memory_example_agent* agent, xmachine_message_example_message_list* example_message_messages){

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int iterated = 0;
    unsigned int count = 0;
    unsigned int sum = 0;
    // Iterate messages
    xmachine_message_example_message *current_message = get_first_example_message_message(example_message_messages);
    while (current_message)
    {
        // If the message is for the same key
        if (agent->key == current_message->key){
            // Increment the counter
            count += 1;
            // Sum the values?
            sum += current_message->value;
        }
        iterated += 1;

        // Get the next message in the list.
        current_message = get_next_example_message_message(current_message, example_message_messages);
    }

    // Print stuff out
    printf("tid %u, key %u, count %u/%u, sum %u\n", tid, agent->key, count, iterated, sum);

    // Set the agents value
    agent->value = count;

    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
