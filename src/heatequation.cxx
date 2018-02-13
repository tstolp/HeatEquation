/**
 * @brief tw3720tu 2017 Final-project heat-equation
 * @authors Jens Langerak 4317327, Thomas Stolp 4377907
 */


#include <iostream>
#include <typeinfo>
#include <math.h>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <ctime>
#include <map>
#include <iterator>
#include <array>

using std::array;

template<typename T, typename S>
bool equals(const T a, const S b)
{
    return false;
}

template<typename T>
bool equals(const T a, const T b)
{
    return a == b;
}

bool equals(const double a, const double b)
{
    return fabs(a - b) < 0.0001;
}


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
     * @brief Create a vector of size n.
     * @param n the size of the vector.
     * @return A vector with size n.
     */
    Vector(const int n)
    : data(new T[n]),
      size(n)
    { }


    /**
     * @brief Create a vector with the elements from the list.
     * @param list values for the vector.
     * @return A vector with the elements from the list.
     */
    Vector(const std::initializer_list<T> list)
    : Vector((int)list.size())
    {
        std::uninitialized_copy(list.begin(), list.end(), data);
    }


    /**
     * @brief Copy constructor.
     * @param v
     * @return A copy of v.
     */
    Vector(const Vector<T>& v)
    : Vector(v.size)
    {
        for (int i=0; i<v.size; i++)
            data[i] = v.data[i];
    }

    /**
     * @brief Move constructor.
     * @param v
     * @return A new vector with the values from v.
     */
    Vector(Vector&& v)
    : data(v.data),
      size(v.size)
    {
        v.size = 0;
        v.data = nullptr;
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


    /**
     * @brief Copy assignment.
     * @param other
     */
    Vector<T>& operator=(const Vector<T>& other)
    {
        if (this != &other)
        {
            delete[] data;
            size = other.size;
            data = new double[other.size];
            for (int i=0; i<other.size; i++)
                data[i] = other.data[i];
            }
        return *this;
    }

    /**
     * @brief Move assignment.
     * @param other
     */
    Vector<T>& operator=(Vector<T>&& other)
    {
        if (this != &other)
        {
            delete[] data;
            size = other.size;
            data = other.data;
            other.size = 0;
            other.data = nullptr;
        }
        return *this;
    }


    /**
     * @brief Add two vectors.
     * @param other
     * @return The sum of this and other.
     */
    Vector<T> operator+(const Vector<T>& other) const
    {
        Vector v(size);
        if(size != other.size)
        {
            throw "Vectors don't have the same size";
        }
        for (auto i=0; i<other.size; i++)
            v.data[i] = data[i] + other.data[i];

        return v;
    }

     /**
     * @brief Subtract two vectors.
     * @param other
     * @return A new vector with the values this-other.
     */
    Vector<T> operator-(const Vector<T>& other) const
    {
        Vector v(size);
        if(size != other.size)
        {
            throw "Vectors don't have the same size";
        }

        for (auto i=0; i<other.size; i++)
             v.data[i] = data[i] - other.data[i];

        return v;
    }

    /**
     * @brief Multiply the vector with a scalar.
     * @param scalar
     * @return A new vector with the values this * scalar.
     */
    Vector<T> operator*(const T scalar) const
    {
        Vector v(size);
        for (auto i=0; i<size; i++)
            v.data[i] = scalar*data[i];
        return v;
    }

    /**
     * @brief Get element at index.
     * @param index index of the requested element.
     * @return A reference to the requested element.
     */
    T& operator[] (const int index) const
    {
        if (index >= size || index < 0)
            throw "Index out of bounds";

        return data[index];
    }

    /**
     * @brief Get the size of the vector.
     * @return The size of the vector.
     */
    int get_size() const
    {
        return size;
    }


    void print() const
    {
        std::cout << "[";
        for(int x = 0; x < size; x++) {
            if (x != 0)
                std::cout << ", ";
            std::cout << data[x];
        }
        std::cout << "]" << std::endl;
    }
};

template <typename T>
bool equals(const Vector<T>& a, const Vector<T>& b)
{
    if (a.get_size() != b.get_size())
        return false;
    for(int i = 0; i < a.get_size(); i++)
    {
        if(!equals(a[i], b[i]))
            return false;
    }

    return true;
}


/**
 * @brief Multiply v with scalar.
 * @param scalar
 * @param v
 * @return A new vector with the values v * scalar;
 */
template <typename T>
Vector<T> operator*(const T scalar, const Vector<T> &v) {
    return v * scalar;
}



/**
* @brief Function that returns the dot product of two vectors.
* @param two vectors of the same length.
* @return the dot product.
*/
template<typename T>
T dot(const Vector<T>& l, const Vector<T>& r)
{
    if(l.get_size() != r.get_size())
    {
        throw "Vectors don't have the same size";
    }

    // Calculate the dot product
    double d=0;
    for (auto i=0; i<l.get_size(); i++)
        d += l[i]*r[i];
    return d;
}

template<typename T>
class Matrix {
private:
    int size_m;
    int size_n;
    std::map<array<int, 2>, T> data;
public:
    /**
     * @brief Create a new matrix with size (m,n).
     * @param m number of rows
     * @param n number of collumns.
     * @return A matrix with size (m,n)
     */
    Matrix(const int m, const int n)
    : size_n(n),
      size_m(m)
    {

    }

    /**
     * @brief Destroy the matrix.
     */
    ~Matrix()
    {
        size_m = 0;
        size_n = 0;
    }

    /**
     * @brief Copy constructor.
     * @param m
     * @return A copy of m.
     */
    Matrix(const Matrix<T>& m)
    : Matrix<T>(m.size_m, m.size_n)
    {
        for (const auto& element : m.data) {
            data[element.first] = element.second;
        }
    }

    /**
     * @brief Move constructor.
     * @param m
     * @return A new matrix with the values of m.
     */
    Matrix(Matrix<T>&& m)
    : size_m(m.size_m),
      size_n(m.size_n)
    {
        m.size_m = 0;
        m.size_n = 0;
        for (const auto& element : m.data) {
            data[element.first] = element.second;
        }
        m.data.clear();
    }

    /**
     * @brief Copy assignment.
     * @param other
     */
    Matrix<T>& operator=(const Matrix<T>& other)
    {
        if (this != &other)
        {
            if (size_m != other.size_m || size_n != other.size_n)
                throw "Wrong size";

            data.clear();
            for (const auto& element : other.data) {
                data[element.first] = element.second;
            }
        }
        return *this;
    }

    /**
     * @brief Move assignment.
     * @param other
     */
    Matrix<T>& operator=(Matrix<T>&& other)
    {
        if (this != &other)
        {
            if (size_m != other.size_m || size_n != other.size_n)
                throw "Wrong size";

            data.clear();
            for (const auto& element : other.data) {
                data[element.first] = element.second;
            }

            other.size_m = 0;
            other.size_n = 0;
            other.data.clear();
        }
        return *this;
    }

    /**
     * @brief Get the element at position index.
     * @param index An array with two values {m,n}.
     * @return A reference to the element at position {m,n}.
     */
    T& operator[] (const array<int, 2> index)
    {
        if (index.size() != 2)
            throw "Needs 2 elements for the index";

        int m = index[0];
        int n = index[1];
        if (m < 0 || n < 0 || m >= size_m || n >= size_n)
            throw "Index out of bounds";

        return data[{m,n}];
    }

    /**
     * @brief Multiply this matrix with vector v.
     * @param v a vector.
     * @return A vector that is the result of this matrix multiplied with v.
     */
    Vector<T> matvec(const Vector<T> &v) const
    {
        Vector<T> result(size_m);
        if (v.get_size() != size_n)
            throw "Wrong size";

        for (int i=0; i< size_m; i++)
            result[i] = 0;

        // use only the non-zero elements from the matrix.
        for (const auto& element : data) {
            result[element.first[0]] += element.second * v[element.first[1]];
        }

        return result;
    }

    /**
     * @brief Multiply this matrix with vector v.
     * @param v a vector.
     * @return A vector that is the result of this matrix multiplied with v.
     */
    Vector<T> operator*(const Vector<T> &v) const
    {
        return matvec(v);
    }

    void print() const
    {
        for(int y = 0; y < size_m; y++) {
            std::cout << "[";
            for(int x = 0; x < size_n; x++) {
                if (x != 0)
                    std::cout << ", ";

                auto search = data.find({y,x});;
                if(search != data.end()) {
                    std::cout << search->second;
                } else {
                    std::cout << 0;
                }
            }
            std::cout << "]" << std::endl;
        }

    }

    /**
     * @brief Returns the size in dimension dim.
     * @param dim 0 or 1.
     * @return The size in the give dimension.
     */
    int get_size(int dim) const
    {
        if (dim == 0) {
            return size_m;
        } else if (dim == 1) {
            return size_n;
        } else {
            throw "Dimension must be 0 or 1";
        }
    }

};

template <typename T>
bool equals(Matrix<T>& a, Matrix<T>& b)
{
    if (a.get_size(0)!= b.get_size(0) || a.get_size(1) != b.get_size(1))
        return false;

    for(int i = 0; i < a.get_size(0); i++)
    {
        for(int j = 0; j < a.get_size(1); j++)
        {
            if(!equals(a[{i,j}], b[{i,j}]))
                return false;
        }
    }

    return true;
}

/**
 * @brief Conjugate Gradient method. Solve Ax = b for x.
 * @param A a symmetric positive definite matrix.
 * @param b
 * @param x initial guess for x. And outputs the result.
 * @param tol the absolute tolerance for the residu.
 * @param maxiter maximum allowed number of iterations.
 * @return number of needed iterations.
 */
template<typename T>
int cg(
    const Matrix<T> &A,
    const Vector<T> &b,
    Vector<T> &x,
    T tol,
    int maxiter)
{
    Vector<T> r = b - A * x;
    Vector<T> p = r;
    Vector<T> A_p;
    for(int k = 0; k < maxiter; k++)
    {
        T dot_r = dot(r,r);

        A_p = A * p;
        T alpha = dot_r / dot(A_p, p);
        x = x + alpha * p;
        r = r - alpha * A_p;
        T dot_r_n = dot(r,r);
        if (dot_r_n < tol*tol) {
            return k;
        }

        T beta = dot_r_n / dot_r;
        p = r + beta * p;
    }
    return -1;
}

class Heat1D
{
private:
    Matrix<double> M;
    Vector<double> u_0;
    double alpha;
    double dt;
    double dx;

public:
    /**
     * @brief Create an object that can solve the heat equation in 1D.
     * @param p_alpha the diffusion coefficient.
     * @param m the number of points per dimension.
     * @param p_dt the timestep.
     */
    Heat1D(double p_alpha, int m, double p_dt) :
        M(m,m),
        alpha(p_alpha),
        dt(p_dt),
        dx( 1.0/(m+1)),
        u_0(m)
    {
        // Calculate matrix M
        for (int i = 0; i < m; i++)
        {
            int l = i - 1;
            int r = i + 1;
            double s = alpha * dt / (dx * dx);
            M[{i,i}] = 1 + 2 * s;
            if (l >= 0)
                M[{i,l}] = -1 * s;
            if (r < m)
                M[{i,r}] = -1 * s;
        }

        // Calculate u_0
        for (int i = 0; i < m; i++)
        {
            u_0[i] = sin(M_PI * (i + 1) * dx); //+1 because cell centered
        }
    }

    /**
     * @brief Calculate the exact solution at time t.
     * @param t
     * @return The exect solution at time t.
     */
    Vector<double> exact(double t) const
    {
        return exp(-M_PI * M_PI * alpha * t) * u_0;
    }

    /**
     * @brief Calculate the numerical solution at time t_end.
     * @param t_end
     * @return The numerical solution at time t_end.
     */
    Vector<double> solve(double t_end) const
    {
        Vector<double> x = u_0;
        int steps = t_end / dt ;

        for (int i = 0; i < steps; i++)
        {
            Vector<double> b = x;
            if (cg<double>(M, b, x, 0.0001, 100) < 0)
                throw "Error";
        }

        return x;
    }
};


class Heat2D
{
private:
    Matrix<double> M;
    Vector<double> u_0;
    double alpha;
    double dt;
    double dx;

public:
    /**
     * @brief Create an object that can solve the heat equation in 2D.
     * @param p_alpha the diffusion coefficient.
     * @param m the number of points per dimension.
     * @param p_dt the timestep.
     */
    Heat2D(double p_alpha, int m, double p_dt) :
        M(m * m,m * m),
        alpha(p_alpha),
        dt(p_dt),
        dx( 1.0/(m+1)),
        u_0(m * m)
    {
        // Calculate matrix M
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                int index = i * m + j;
                int d = i - 1;
                int u = i + 1;
                int l = j - 1;
                int r = j + 1;
                double s = alpha * dt / (dx * dx);
                M[{index,index}] = 1 + 2 * 2 * s;
                if (l >= 0)
                    M[{index,index - 1}] = -1 * s;
                if (r < m)
                    M[{index,index + 1}] = -1 * s;
                if (d >= 0)
                    M[{index,index - m}] = -1 * s;
                if (u < m)
                    M[{index,index + m}] = -1 * s;
            }
        }

        // Calculate u_0
        for (int i = 0; i < m; i++)
        {
            double u_i = sin(M_PI * (i + 1) * dx); //+1 because cell centered
            for (int j = 0; j < m; j++)
            {
                u_0[i * m + j] = u_i + sin(M_PI * (j + 1) * dx); //+1 because cell centered
            }
        }
    }

    /**
     * @brief Calculate the exact solution at time t.
     * @param t
     * @return The exect solution at time t.
     */
    Vector<double> exact(double t) const
    {
        return exp(-2*M_PI * M_PI * alpha * t) * u_0;
    }

    /**
     * @brief Calculate the numerical solution at time t_end.
     * @param t_end
     * @return The numerical solution at time t_end.
     */
    Vector<double> solve(double t_end) const
    {

        Vector<double> x = u_0;
        int steps = t_end / dt ;

        for (int i = 0; i < steps; i++)
        {
            Vector<double> b = x;
            if (cg<double>(M, b, x, 0.01, 1000) < 0)
                throw "Error";
        }

        return x;
    }
};

template<int n> class Heat
{
private:
    Matrix<double> M;
    Vector<double> u_0;
    double alpha;
    double dt;
    double dx;

public:
    /**
     * @brief Create an object that can solve the heat equation in nD.
     * @param p_alpha the diffusion coefficient.
     * @param m the number of points per dimension.
     * @param p_dt the timestep.
     */
    Heat(double p_alpha, int m, double p_dt) :
        M(pow(m,n), pow(m,n)),
        alpha(p_alpha),
        dt(p_dt),
        dx( 1.0/(m+1)),
        u_0(pow(m,n))
    {
        int points = pow(m,n);
        int x[n] {0}; //keep track of the coordinate
        double s = alpha * dt / (dx * dx);

        // iterate over all the point while keeping track of the coordinates of the points.
        for (int i = 0; i < points; i++)
        {
            // calculate u_0
            u_0[i] = 0;
            for (int j = 0; j < n; j++)
            {
                u_0[i] += sin(M_PI * (x[j] + 1) * dx);
            }


            // calculate row i in matrix M.
            M[{i,i}] = 1 + 2 * n * s;
            for (int j = 0; j < n; j++) {
                int index_l = i - pow(m, j);
                int index_h = i + pow(m, j);
                if (x[j] - 1 >= 0)
                    M[{i,index_l}] = -1 * s;
                if (x[j] + 1 < m)
                    M[{i,index_h}] = -1 * s;
            }

            // update coordinate.
            for (int j = 0; j < n; j++)
            {
                x[j]++;
                if (x[j] == m)
                    x[j] = 0;
                else
                    break;
            }
        }
    }

    /**
     * @brief Calculate the exact solution at time t.
     * @param t
     * @return The exect solution at time t.
     */
    Vector<double> exact(double t) const
    {
        return exp(-n*M_PI * M_PI * alpha * t) * u_0;
    }

    /**
     * @brief Calculate the numerical solution at time t_end.
     * @param t_end
     * @return The numerical solution at time t_end.
     */
    Vector<double> solve(double t_end) const
    {
        Vector<double> x = u_0;
        int steps = t_end / dt ;

        for (int i = 0; i < steps; i++)
        {
            Vector<double> b = x;
            if (cg<double>(M, b, x, 0.01, 1000) < 0)
                throw "Error";
        }

        return x;
    }
};





/**************************************************************
 * TEST CODE                                                  *
 **************************************************************/
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
    Vector<double> J({1,2,3.1});
    Vector<double> K({0.9,2,3});
    Vector<double> L = I;
    assert(A.get_size() == 0);
    assert(B.get_size() == 0);
    assert(C.get_size() == 0);
    assert(F.get_size() == 1);
    assert(H.get_size() == 3);
    assert(equals(A,B));
    assert(equals(A,C));
    assert(!equals(A,E));

    assert(!equals(A,F));
    assert(!equals(A,G));
    assert(!equals(F,G));
    assert(!equals(G,H));
    assert(equals(H,I));
    assert(!equals(I,J));
    assert(!equals(I,K));
    assert(equals(L,I));
    assert(&L != &I);
    I[0] = 0.9;
    assert(!equals(L,I));
    assert(equals(I,K));
}

void test_vector_operations(void)
{
    Vector<double> A({1,2,3});
    Vector<double> B({2,4,-1});

    Vector<double> expected({3,6,2});
    assert(equals(expected, A+B));

    expected = Vector<double>({-1,-2,4});
    assert(equals(expected, A-B));

    expected = Vector<double>({4, 8, -2});
    assert(equals(expected, 2.0 * B));
    assert(equals(expected, B * 2.0));

    assert(equals(7.0, dot(A,B)));

    bool exception = false;
    Vector<double> C({2,4});
    try {
        A+C;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;

    try {
        C+A;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;

    try {
        A-C;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;

    try {
        C-A;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;

    try {
        dot(A,C);
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;

    try {
        dot(C,A);
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
}

void test_matrix(void)
{
    Matrix<double> test(3,4);
    assert(equals(3, test.get_size(0)));
    assert(equals(4, test.get_size(1)));
    assert(equals(0.0, test[{1,2}]));
    assert(equals(0.0, test[{0,0}]));
    test[{0,0}] = -1.1;
    test[{2,3}] = 1.0;
    assert(equals(-1.1, test[{0,0}]));
    assert(equals(1.0, test[{2,3}]));

    bool exception = false;
    try {
        test[{3,0}] = 1.0;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
    exception = false;

    try {
        test[{0,4}] = 1.0;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);
}

void test_matrix_vector(void)
{
    Matrix<double> M(3,4);
    Vector<double> v({1,2,3,4});
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            M[{i,j}] = i + j;
        }
    }

    Vector<double> expected({20, 30, 40});

    assert(equals(expected, M*v));

    Vector<double> v2({1,2,3});
    bool exception = false;
    try {
        M*v2;
    } catch (char const* e) {
        exception = true;
    }
        assert(exception);

    exception = false;
    Vector<double> v3({1,2,1,1,3});
    try {
        M*v3;
    } catch (char const* e) {
        exception = true;
    }
    assert(exception);

}

/**
 * @brief Test the code.
 */
void test(void)
{
    test_vector_constructor();
    test_vector_operations();
    test_matrix();
    test_matrix_vector();
    std::cout << "All test passed!" << std::endl;
}

template<typename T>
T error(Vector<T> a, Vector<T> b)
{
    T result;
    for (int i = 0; i < a.get_size(); i++)
    {
        result += fabs(a[i] - b[i]);
    }
    return result;

}

int main(){
    test();

    Heat1D test(0.3125, 99, 0.0001);
    try {
    Vector<double> a = test.exact(1);

    std::cout << "head1D, start solving ..." << std::endl;
    Vector<double> b = test.solve(1);

    std::cout << "total error: " << error(a,b) << std::endl;
    std::cout << "average error: " << error(a,b) / 99 << std::endl;


    Heat2D test2(0.3125, 99, 0.001);
    Vector<double> a2 = test2.exact(1);

    std::cout << "head2D, start solving ..."  << std::endl;
    Vector<double> b2 = test2.solve(1);

    std::cout << "total error: " << error(a2,b2) << std::endl;
    std::cout << "average error: " << error(a2,b2) / (99*99) << std::endl;

    Heat<1> test3(0.3125, 99, 0.001);
    Vector<double> a3 = test3.exact(1);

    std::cout << "head<1>, start solving ..."  << std::endl;
    Vector<double> b3 = test3.solve(1);

    std::cout << "total error: " << error(a3,b3) << std::endl;
    std::cout << "average error: " << error(a3,b3) / 99 << std::endl;

    std::cout << "difference between head<1> and head1D (exact): " << error(a,a3) << std::endl;
    std::cout << "difference between head<1> and head1D (solve): " << error(b,b3) << std::endl;


    Heat<2> test4(0.3125, 99, 0.001);
    Vector<double> a4= test4.exact(1);

    std::cout << "head<2>, start solving ..."  << std::endl;
    Vector<double> b4 = test4.solve(1);

    std::cout << "total error: " << error(a4,b4) << std::endl;
    std::cout << "average error: " << error(a4,b4) / (99*99) << std::endl;

    std::cout << "difference between head<2> and head2D (exact): " << error(a2,a4) << std::endl;
    std::cout << "difference between head<2> and head2D (solve): " << error(b2,b4) << std::endl;

    Heat<3> test5(0.3125, 99, 0.001);
    Vector<double> a5 = test5.exact(1);

    std::cout << "head<3>, start solving ..."  << std::endl;
    Vector<double> b5 = test5.solve(1);

    std::cout << "total error: " << error(a5,b5) << std::endl;
    std::cout << "average error: " << error(a5,b5) / (99*99*99) << std::endl;
    } catch(const char * e) {
            std::cout << e << std::endl;
    }


    return 0;
}
