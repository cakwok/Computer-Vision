//
//  main.cpp
//  pow.cpp
//
//  Created by Casca on 28/2/2022.
//

#include <iostream>
#include <chrono>  // for high_resolution_clock
#include <math.h>

int main(int argc, const char * argv[]) {

    float power = 0.0f;
    
    auto start = std::chrono::high_resolution_clock::now(); // Record start time
    
    // record time used by pow
    for (int i = 0; i<100; i++) {
        power = pow(0.002342342342808909354,2);
    }

    auto finish = std::chrono::high_resolution_clock::now();   // Record end time
    
    std::chrono::duration<double> total_time_taken = finish - start;     
    std::cout << "Time taken pow(0.002342342342808909354,2): " << total_time_taken.count() << "\n" ;
   
    start = std::chrono::high_resolution_clock::now(); // Record start time
    
    // record time used by multiplication
    for (int i = 0; i<100; i++) {
        power = 0.002342342342808909354 * 0.002342342342808909354;
    }

    finish = std::chrono::high_resolution_clock::now();   // Record end time
    
    total_time_taken = finish - start;     //cout output
    std::cout << "Time taken (Multiplication, power = 0.002342342342808909354 * 0.002342342342808909354)" << total_time_taken.count() << "\n" ;
    
    return 0;
    
}
