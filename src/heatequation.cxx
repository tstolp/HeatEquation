#include <iostream>
#include <typeinfo>
#include <math.h>
#include <cassert>
#include <initializer_list>
#include <memory>


template<typename T>
class Vector {
private:
    T* data;
    int size;

public:

    /**
     * @brief Default constructor for a vector.
     * @return An vector with size 0.
     */
    Vector()
    : size(0),
      data(nullptr)
    { }

    /**
     * @brief Create a vector of size n
     * @param n the size of the vector.
     * @return A vector with size n.
     */
    Vector(int n)
    : data(new T[n]),
      size(n)
    { }

    /**
     * @brief Create a vector and initialize the vector with the values in the list.
     * @param list the values for the vector.
     * @return A vector initialized with the values from the list.
     */
    Vector(std::initializer_list<T> list)
    : Vector((int)list.size())
    {
        std::uninitialized_copy(list.begin(), list.end(), data);
    }

    Vector(const Vector<T>& v)
    : Vector(v.size)
    {
        for (int i=0; i<v.size; i++)
            data[i] = v.data[i];
    }

    /**
     * @brief Destroy the vector.
     * @return
     */
    ~Vector()
    {
        size=0;
        delete[] data;
    }


    template<typename A>
    bool equals(Vector<A> other)
    {
        return false;
    }

    bool equals(Vector<T> other)
    {
        if (this->size != other.size)
            return false;
        for(int i = 0; i < this->size; i++)
        {
            if(!fabs(this->data[i] - other.data[i]) < 0.0001)
                return false;
        }

        return true;
    }

      // Copy assignment
    template<typename T>
    Vector<T>& operator=(const Vector<T>& other<T>)
    {
        if (this != &other)
            {
                delete[] data;
                size = other.size;
                data   = new double(other.size);
                for (auto i=0; i<other.size; i++)
                    data[i] = other.data[i];
            }
        return *this;
    }

    template<typename T>
    Vector operator+(const Vector& other) const
    {
        Vector v(size);
        if(size != other.size)
        {
            std::cout<<"Vectors don't have the same size" << std::endl;
        }
        else{
            for (auto i=0; i<other.size; i++)
                    v.data[i] = data[i] + other.data[i];
        }

        std::clog << "plus operator" << std::endl;
        return v;
    }

    template<typename T>
    Vector operator-(const Vector& other) const
    {
        Vector v(size);
        if(size != other.size)
        {
            std::cout<<"Vectors don't have the same size" << std::endl;
        }
        else{
            for (auto i=0; i<other.size; i++)
                    v.data[i] = data[i] - other.data[i];
        }

        std::clog << "minus operator" << std::endl;
        return v;
    }
};
};


/**
* @brief Function that returns the dot product of two vectors.
* @param two vectors of the same length.
* @return the dot product.
*/

template<typename T>
T dot(const Vector<T>& l, const Vector<T>& r)
{
    // Calculate the dot product
    double d=0;
    for (auto i=0; i<l.size; i++)
        d += l.data[i]*r.data[i];
    return d;
}

void test_vector_constructor(void)
{
    Vector<double> A;
    Vector<double> B(0);
    Vector<double> C({});
    Vector<int> E;
    Vector<double> F(1);
    Vector<double> G({1});
    Vector<double> H({1,2,3});
    Vector<double> I({1,2,3});
    Vector<double> J({1,2,0});
    Vector<double> K({0,2,3});
    Vector<double> L = I;
    assert(A.equals(B));
    assert(A.equals(C));
    assert(!A.equals(E));
    assert(!A.equals(F));
    assert(!A.equals(G));
    assert(!F.equals(G));
    assert(!G.equals(H));
    assert(H.equals(I));
    assert(!I.equals(J));
    assert(!I.equals(K));
    assert(L.equals(I));
    assert(&L != &I);
    // std::cout << dot(H, I) << std::endl;
}

/**
 * @brief Test the code.
 */
void test(void)
{
    test_vector_constructor();

    std::cout << "All test passed!" << std::endl;
}

int main(){
    test();

    return 0;
}
