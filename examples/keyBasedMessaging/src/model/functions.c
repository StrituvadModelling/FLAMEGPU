
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

// Output a message
__FLAME_GPU_FUNC__ int output_example(xmachine_memory_agent* agent, xmachine_message_msg_list* msg_messages){

    // 0 output a message.
    add_msg_message(msg_messages, agent->id, agent->key, agent->value);

    return 0;
}

// Iterate messages, counting how many were read and how many of them were relevant.
#if defined(xmachine_message_msg_partitioningNone)
__FLAME_GPU_FUNC__ int input_example(xmachine_memory_agent *agent, xmachine_message_msg_list *msg_messages)
#endif
#if defined(xmachine_message_msg_partitioningKeyBased)
__FLAME_GPU_FUNC__ int input_example(xmachine_memory_agent *agent, xmachine_message_msg_list *msg_messages, xmachine_message_msg_bounds *message_bounds)
#endif
{

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int iterated = 0;
    unsigned int count = 0;
    unsigned int sum = 0;
    // Iterate messages
    #if defined(xmachine_message_msg_partitioningNone)
    xmachine_message_msg *current_message = get_first_msg_message(msg_messages);
    #elif defined(xmachine_message_msg_partitioningKeyBased)
    xmachine_message_msg *current_message = get_first_msg_message(msg_messages, message_bounds, agent->key);
    #endif
    while (current_message)
    {
        // If the message is for the same key
        if (agent->key == current_message->key)
        {
            // Increment the counter
            count += 1;
            // Sum the values?
            sum += current_message->value;
        }
        iterated += 1;

        // Get the next message in the list.
    #if defined(xmachine_message_msg_partitioningNone)
        current_message = get_next_msg_message(current_message, msg_messages);
    #elif defined(xmachine_message_msg_partitioningKeyBased)
        current_message = get_next_msg_message(current_message, msg_messages, message_bounds);
    #endif
    }

    // Print stuff out
    printf("tid %u, key %u, count %u/%u, sum %u\n", tid, agent->key, count, iterated, sum);

    // Set the agents value
    agent->value = count;

    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
