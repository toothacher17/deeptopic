#include <iostream>
#include "digamma.h"


// x would be greater than 0, most case greater than 1
float ApproxDigamma(float x) {
    if (x == 0) {
        x = 0.0001;
    }
    float xp2 = x+2;

    // log(xp2) - (6x+13)/(12*xp2*xp2) - (2x+1)/(x*x+x)
    return std::log(xp2) - (6*x+13)/(12*xp2*xp2) - (2*x+1)/(x*x+x);
}

int main() {
    std::cout << ApproxDigamma(0) << std::endl;
    std::cout << ApproxDigamma(0.001) << std::endl;
    std::cout << ApproxDigamma(0.1) << std::endl;
    std::cout << ApproxDigamma(2) << std::endl;

    return 0;
}

