#include <iostream>
#include <typeinfo>
#include <math.h>
#include <cassert>

/**
 * @brief Test if expected and actual are equal.
 * @param expected The expected value.
 * @param actual The actual value.
 * @return True if expexted equals the actual value.
 */
bool equals(double expected, double actual)
{
    return fabs(expected - actual) < 0.0001;
}

/**
 * @brief Test the code.
 */
void test(void)
{
    //test equals
    assert(equals(2.0, 2.0));
    assert(!equals(2.0001, 2.00021));
    
    
    
    
    std::cout << "All test passed!" << std::endl;
}

int main(){
    test();

    return 0;
}
