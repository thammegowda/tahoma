#include <iostream>
#include <rtg.hpp>

// rename main as trainer_main 
#define main trainer_main
#include "nmt/trainer.hpp"
#undef main


int main(int argc, char* argv[]) {
    using namespace rtg;
    int _code = global_setup();
    if (_code != 0){
        return _code;
    }
    return trainer_main(argc, argv);
}